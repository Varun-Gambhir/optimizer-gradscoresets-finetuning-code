from absl import app
import jax
from tunix.generate import sampler
from tunix.cli import config
from tunix.sft import checkpoint_manager
from tunix.cli.utils import model as model_lib
import numpy as np
import flax.nnx as nnx
# from eval.mmlu.categories import categories, subcategories
from tunix.sft.eval.data_selection.get_validation_dataset import get_mmlu_dataset_df
from tunix.sft.eval.eval_utils import get_next_word_predictions
import tunix.sft.eval.chat_templates as chat_templates
import tunix.sft.eval.colm.run_eval as colm_eval
import os, json
import datasets

from tqdm import tqdm

choices = ["A", "B", "C", "D"]



'''
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"  # or "Qwen/Qwen3-7B-Instruct" when available
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Your task-specific instruction
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers multiple-choice science questions. "
    "Always answer with the letter (A, B, C, or D) only."
)

# Few-shot examples as (input, output) pairs
few_shot_examples = [
    {
        "input": "Q: What is the chemical symbol for water?\n"
                 "A. O2\nB. CO2\nC. H2O\nD. HO\n",
        "output": "C",
    },
    {
        "input": "Q: Which planet is known as the Red Planet?\n"
                 "A. Venus\nB. Mars\nC. Jupiter\nD. Saturn\n",
        "output": "B",
    },
    # ... more shots if you want
]

def build_few_shot_messages(question_text: str, few_shots):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add each few-shot pair as its own user/assistant turns
    for ex in few_shots:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})

    # Final test query as the last user turn
    messages.append({"role": "user", "content": question_text})

    return messages

# Example test question
test_question = (
    "Q: What gas do plants primarily absorb for photosynthesis?\n"
    "A. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Hydrogen\n"
)

messages = build_few_shot_messages(test_question, few_shot_examples)

# This is the *standard* HF way as of recent Transformers versions
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # adds the assistant prefix per Qwen's chat template
)


'''

def format_example(df, idx, include_answer):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nThe answer is:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt



def get_qa_pair(df, idx):
    """
    Returns:
        question_str: stem + options + 'The answer is:' (no answer filled in)
        answer_str:   the correct answer token from the df (e.g. 'A', 'B', ...)
    """
    # stem
    question = df.iloc[idx, 0]

    # number of options = total cols - stem col - answer col
    k = df.shape[1] - 2

    # append options
    for j in range(k):
        question += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])

    # end with the prompt for the model
    question += "\nThe answer is:"

    # gold answer (last column)
    answer = df.iloc[idx, k + 1]

    return question, answer

def get_sysprompt(subject):
    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
    
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    return prompt

def gen_prompt(df, subject, k=-1):

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
    
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = df.shape[0]
    for i in range(k):
        prompt += format_example(df, i, include_answer=True)
    return prompt


def eval_hf_model_generate_ICL_prompts(args, tokenizer, dev_df, test_df):
    subject = args["subject"]
    k0 = int(args["n_val"])
    max_len = int(args["max_prompt_tokens"])

    # chat_formatting_function = create_prompt_with_tulu_chat_format
    
    prompts = []
    skipped_lengths = []

    def add_answer_trigger(p: str) -> str:
        if p and p[-1] in ["\n", " "]:
            return p + "The answer is:"
        return p + " The answer is:"

    def process(msg, add_bos=False):
        txt = templatize([msg], tokenizer)[0]
        # txt = chat_formatting_function([{"role": "user", "content": msg}], add_bos=add_bos)
        txt = add_answer_trigger(txt)
        tokens = tokenizer(txt, truncation=False, add_special_tokens=False).input_ids
        return txt

    def tok(msg): return tokenizer(msg, truncation=False, add_special_tokens=False).input_ids

    for i in range(test_df.shape[0]):
        k = k0
        prompt_end = format_example(test_df, i, include_answer=False)

        # First check: if test example alone doesn't fit, skip it.
        base = process(prompt_end)
        base_len = len(tok(base))
        if base_len > max_len:
            skipped_lengths.append(base_len)
            continue

        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = process(train_prompt + prompt_end)
        tokenized_prompt = tok(prompt)
        while len(tokenized_prompt) > max_len and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = process(train_prompt + prompt_end)
            tokenized_prompt = tok(prompt)

        # If it still doesn't fit even with k=0, skip it (rare, but possible due to formatting).
        if len(tokenized_prompt) > max_len:
            skipped_lengths.append(len(tokenized_prompt))
            continue

        prompts.append(prompt)
    if len(skipped_lengths):
      print(f"[ICL] skipped={len(skipped_lengths)} / {test_df.shape[0]}")
      print(f"[ICL] skipped_lengths={skipped_lengths}")

    return prompts

def compute_accuracy_mmlu(args, inference_function, tokenizer):
    ddir = "/home/temp/data"
    dev_df = get_mmlu_dataset_df(data_dir=ddir,
                                 validation=True, 
                                 k = args["n_val"], 
                                 subject = args["subject"])

    test_df = get_mmlu_dataset_df(data_dir=ddir,
                                 validation=False, 
                                 k = args["n_test"], 
                                 subject = args["subject"])
    prompts = eval_hf_model_generate_ICL_prompts(args, tokenizer, dev_df, test_df)
    pred_indices = inference_function(
        prompts,
    )[1]

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    return cors, acc

def templatize(prompts, tokenizer):
  out = []
  for p in prompts:
    out.append(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
  return out

class PeftPipeline(config.HyperParameters):

  def get_model(self, load_fresh=False):
    mesh: jax.sharding.Mesh = self.create_mesh('model_config')
    print("config", self.config["training_config"])
    ckpt_root = self.config["training_config"]["checkpoint_root_directory"]
    
    model, tokenizer_path = model_lib.create_model(
      self.config['model_config'], self.config['tokenizer_config'], mesh
    )
    if model is None:
      raise ValueError('model is None')
    tokenizer = model_lib.create_tokenizer(
      self.config['tokenizer_config'], tokenizer_path
    )
    if not load_fresh:
        print("LOADING FROM CHECKPOINT")
        ckpt_manager = checkpoint_manager.CheckpointManager(
            root_directory=ckpt_root
        )
        ckpt_manager.maybe_restore(model, self.config["inference_restore_step"], restore_only_lora_params=True)
    return model, mesh, tokenizer

  def run_inference(self, my_sampler,max_generation_steps, 
                    max_prompt_length, candidate_token_ids, batch_size, inputs):
    
    # inputs = templatize([
    #   "which is larger 9.9 or 9.11?",
    #   "What is the capital of France?\nA) Berlin\nB) Madrid\nC) Paris\nD) Rome"
    #   "What is the capital of France?\nA) Berlin\nB) Madrid\nC) Paris\nD) Rome"
    # ], tokenizer._tokenizer)
    mysamp = my_sampler
    
    N = len(inputs)
    if N == 0: raise ValueError("inputs is empty")

    num_batches = (N + batch_size - 1) // batch_size  # ceil(N / batch_size)
    pad = num_batches * batch_size - N  # handles N < batch_size too

    # Pad by repeating real examples (cycle through inputs to avoid always repeating last)
    inputs_padded = inputs
    logits_chunks = []
    out_strs = []
    PROC = max_generation_steps == 1

    if pad > 0:
      reps = (pad + N - 1) // N  # enough repeats so inputs * reps has >= pad elems
      inputs_padded = inputs + (inputs * reps)[:pad]

    for i in range(0, len(inputs_padded), batch_size):
      batch = inputs_padded[i : i + batch_size]  # always length == batch_size
      outs = mysamp(
          batch,
          max_generation_steps=max_generation_steps,
          echo=False,
          return_logits=True,
          max_prompt_length=max_prompt_length,
      )

      out_strs.extend(outs.text)
      if PROC: logits_chunks.extend(outs.logits)

    if PROC:
        batch_logits = jax.numpy.concat(logits_chunks, axis=0)[:N]
        batch_probs = jax.scipy.special.softmax(batch_logits, axis=-1)
        print("lens", len(inputs), len(inputs_padded), batch_probs.shape)

        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]

        batch_prediction_indices = jax.numpy.argmax(batch_probs, axis=-1)

        return out_strs, batch_prediction_indices.tolist()
    return out_strs, None


from functools import partial
from pathlib import Path
def get_subjects(folder_path):
    names = []
    for p in Path(folder_path).iterdir():
        if p.is_file():
            name = p.stem
            if name.endswith("_dev"):
                name = name[:-4]  # remove '_dev'
            names.append(name)
    return names

def main(argv, **kwargs):
  COLM_eval(argv, **kwargs)
#   MOL_eval(argv, **kwargs)


def COLM_eval(argv, **kwargs):
  pipeline = PeftPipeline(argv, **kwargs)
  model, mesh, tokenizer = pipeline.get_model(load_fresh=False)
  exp_name = pipeline.config["training_config"]["checkpoint_root_directory"]
  ckpt_num = pipeline.config["inference_restore_step"]
  exp_name = os.path.basename(exp_name.rstrip("/"))
  tokenizer._tokenizer.chat_template = chat_templates.qwen2_5_template

  base_args = {
    "max_prompt_tokens": 1024,
    "max_generation_steps": 256,
  }
  batch_size = 8

  out_dir = f"/home/temp/tunix/examples/sft/mtnt/results/{exp_name}/{ckpt_num}"
  data_root = "/home/temp/CoLM/math_eval/dataset"
  os.makedirs(out_dir, exist_ok=True)
  
  with mesh:
    mysamp = sampler.Sampler(
        model,
        tokenizer,
        sampler.CacheConfig(
            cache_size=2048,
            num_layers=model.config.num_layers,
            num_kv_heads=model.config.num_kv_heads,
            head_dim=model.config.head_dim,
        ),
    )
    
    inference_function = partial(pipeline.run_inference, mysamp, base_args["max_generation_steps"], 
                                 base_args["max_prompt_tokens"], [], batch_size)
    inf2 = lambda batch: inference_function(batch)[0]
    # subjects = get_subjects("/home/temp/data/eval/mmlu/dev")
    # colm_dataset = ["gsm8k", "numglue", "svamp"]
    # colm_dataset = ["math", "gsm8k", "numglue",  "svamp", "mmlu_mathematics", "aqua", "simuleq", "sat"]
    maths = [
        'mmlu_elementary-mathematics', 
        'mmlu_high-school-mathematics', 
        'mmlu_college-mathematics', 
        'mmlu_abstract-algebra', 
        'mmlu_formal-logic']
    numglue = [
        'numglue_Type_2', 'numglue_Type_4', 'numglue_Type_3', 'numglue_Type_8', 'numglue_Type_1'
    ]
    # colm_dataset = ["math", "gsm8k",  "svamp", "simuleq", "deepmind"]
    colm_dataset = ["numglue", "mmlu_mathematics", "gsm8k",  "svamp", "simuleq", "deepmind", "aqua", "sat"]
    # colm_dataset = ["gsm8k", "mmlu_mathematics"]
    # colm_dataset = ["gsm8k"]
    # colm_dataset = ["numglue", "svamp"]
    flan_tag = ""
    all_res = []
    for eval_ds in colm_dataset:
      num_shots = 0
      if "mmlu" in eval_ds: num_shots = 2
      if "aqua" in eval_ds: num_shots = 2
      if "sat" in eval_ds: num_shots = 2
      print(f"TESTING {eval_ds} at {num_shots} shots")
      fname = f"{eval_ds}_acc.jsonl"
      if "pot" in flan_tag: fname = f"{eval_ds}_pot_acc.jsonl"
      out_path = os.path.join(out_dir, fname)
      res = colm_eval.run_eval(inf2, tokenizer._tokenizer, 
                               base_args["max_prompt_tokens"], out_path, data_root, eval_ds, num_shots, 
              batch_size, stem_flan_type=flan_tag, cot_backup=True, debug=True)
      all_res.append(res)

  summarize_results(all_res, f"{out_dir}/results.csv")

'''

instead of appending all res. "update" existing results. accuracy must be updated. two results are mergeable only when dataset and shots are same. if not, add a new entry. hope this makes sense and is obvious. No emojis in chat. also, instead of csv, write in json instead as i think it might be simpler. but dont add too much indentation which printing to file. want my file to be concise. but not unreadable at the same time
'''
def summarize_results(results, csv_path):
    import os
    from tabulate import tabulate
    import csv
    from collections import defaultdict

    # Overall average
    avg_acc = sum(r["accuracy"] for r in results) / len(results)

    # Compute average grouped by shots
    grouped = defaultdict(list)
    for r in results:
        grouped[r["shots"]].append(r["accuracy"])
    group_avgs = {shots: sum(vals) / len(vals) for shots, vals in grouped.items()}

    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        fieldnames = ["dataset", "accuracy", "shots"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        # Original rows
        writer.writerows(results)

        # Group averages
        for shots, avg in sorted(group_avgs.items(), key=lambda x: x[0]):
            writer.writerow({"dataset": f"AVERAGE_SHOT_{shots}", "accuracy": avg, "shots": shots})

        # Overall average
        writer.writerow({"dataset": "AVERAGE", "accuracy": avg_acc, "shots": ""})

    # Display
    display_rows = results.copy()
    for shots, avg in sorted(group_avgs.items(), key=lambda x: x[0]):
        display_rows.append({"dataset": f"AVERAGE_SHOT_{shots}", "accuracy": avg, "shots": shots})
    display_rows.append({"dataset": "AVERAGE", "accuracy": avg_acc, "shots": ""})

    print("\nAverage Accuracy: {:.4f}\n".format(avg_acc))
    print(tabulate(display_rows, headers="keys", tablefmt="github", floatfmt=".4f"))
    print(csv_path)

    return avg_acc, group_avgs



def MOL_eval(argv, **kwargs):
  pipeline = PeftPipeline(argv, **kwargs)
  model, mesh, tokenizer = pipeline.get_model(load_fresh=False)
  exp_name = pipeline.config["training_config"]["checkpoint_root_directory"]
  ckpt_num = pipeline.config["inference_restore_step"]
  exp_name = os.path.basename(exp_name.rstrip("/"))
  tokenizer._tokenizer.chat_template = chat_templates.qwen2_5_template
  tok = tokenizer._tokenizer
  base_args = {
    "max_prompt_tokens": 896,
    "max_generation_steps": 128,
  }
  num_test = 1000
  batch_size = 16
  root = "/home/temp/tunix/examples/sft/mtnt/results"
  out_dir = f"{root}/{exp_name}/{ckpt_num}/MOL_{num_test}"
  os.makedirs(out_dir, exist_ok=True)
  
  with mesh:
    mysamp = sampler.Sampler(
        model,
        tokenizer,
        sampler.CacheConfig(
            cache_size=1024,
            num_layers=model.config.num_layers,
            num_kv_heads=model.config.num_kv_heads,
            head_dim=model.config.head_dim,
        ),
    )
    
    inference_function = partial(pipeline.run_inference, mysamp, base_args["max_generation_steps"], 
                                 base_args["max_prompt_tokens"], [], batch_size)
    inf2 = lambda batch: inference_function(batch)[0]
    MOL_GEN(inf2, num_test, tok, out_dir, batch_size=batch_size)
    import tunix.sft.eval.mol.molecule.evaluate as mol_eval
    mol_eval.run_eval(out_dir)
    

# df = pd.read_csv(
#     "file.tsv",
#     sep="\t",
#     quoting=csv.QUOTE_ALL,
#     escapechar="\\"
# )
# df = pd.read_csv(
#     "file.tsv",
#     sep="\t",
#     quoting=csv.QUOTE_ALL,   # respect quoting
#     escapechar="\\",         # match writer
#     engine="python"          # <-- important for multiline support
# )
def MOL_GEN(inference_fn, num_test, tokenizer, out_dir, batch_size=32):
    os.makedirs(out_dir, exist_ok=True)

    cfgs = ['Molecule-oriented Instructions', 'Protein-oriented Instructions', 'Biomolecular Text Instructions']
    ds = datasets.load_dataset("zjunlp/Mol-Instructions", cfgs[0], trust_remote_code=True)

    keys = [
        "description_guided_molecule_design",
        "forward_reaction_prediction",
        "reagent_prediction",
        "retrosynthesis"
    ]
    dsdict = {}
    dsdict_demos = {}
    n_shots = 2

    for i,key in enumerate(keys):
        t = ds[key].train_test_split(test_size=num_test, seed=42, shuffle=True)
        dsdict[key] = t["test"]
        dsdict_demos[key] = {"instruction": [], "input": [], "output": []}
        if n_shots > 0:
          dsdict_demos[key] = t["train"].train_test_split(test_size=n_shots, seed=42, shuffle=True)["test"]

    for key, _ in dsdict.items():
      dataset = dsdict[key]
      demos = dsdict_demos[key]
      print("PROCESSING", key, len(dataset), len(dataset) // batch_size)
      out_path = f"{out_dir}/{key}.jsonl"

      with open(out_path, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(dataset), batch_size), desc=key):
          batch = dataset[start:start+batch_size]
          prompts = chat_templates.promptify_alpaca([], batch)

          ground = batch["output"]
          preds = inference_fn(prompts)

          for d, g, p in zip(prompts, ground, preds):
            record = {
              "description": d,
              "ground_truth": g,
              "output": p,
            }
            # ensure_ascii=False keeps Unicode, 
            # JSON escapes newlines/quotes automatically
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



def MMLU_eval(argv, **kwargs):
  pipeline = PeftPipeline(argv, **kwargs)
  model, mesh, tokenizer = pipeline.get_model()
  exp_name = pipeline.config["training_config"]["checkpoint_root_directory"]
  ckpt_num = pipeline.config["inference_restore_step"]
  exp_name = os.path.basename(exp_name.rstrip("/"))
  choices = ["A", "B", "C", "D"]
  tokenizer._tokenizer.chat_template = chat_templates.qwen2_5_template
  candidate_token_ids = [
    tokenizer._tokenizer.encode(" " + c, add_special_tokens=False)[-1]
    for c in choices
  ]

  base_args = {
    "n_val": 1,
    "n_test": 2000,
    "max_prompt_tokens": 512,
    "max_generation_steps": 1,
  }

  out_dir = f"/home/temp/tunix/examples/sft/mtnt/results/{exp_name}/{ckpt_num}"
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, "mmlu_acc.jsonl")

  with mesh:
    mysamp = sampler.Sampler(
        model,
        tokenizer,
        sampler.CacheConfig(
            cache_size=2048,
            num_layers=model.config.num_layers,
            num_kv_heads=model.config.num_kv_heads,
            head_dim=model.config.head_dim,
        ),
    )
    batch_size = 32
    inference_function = partial(pipeline.run_inference, mysamp, base_args["max_generation_steps"], 
                                 base_args["max_prompt_tokens"], candidate_token_ids, batch_size)
    subjects =  [
      "abstract_algebra",
      "college_chemistry",
      "college_computer_science",
      "college_mathematics",
      "college_physics",
      "conceptual_physics",
      "electrical_engineering",
      "elementary_mathematics",
      "formal_logic",
      "high_school_chemistry",
      "high_school_computer_science",
      "high_school_mathematics",
      "high_school_physics",
      "high_school_statistics"
    ]

    acc_list = []

    with open(out_path, "a") as f:
      f.write(json.dumps({"type": "meta", **base_args}) + "\n")
      f.flush()

      for sub in subjects:
        args = dict(base_args)
        args["subject"] = sub

        acc = compute_accuracy_mmlu(args, inference_function, tokenizer._tokenizer)[1]
        print("Accuracy", sub, acc)

        acc_list.append(acc)
        curr_mean = sum(acc_list) / len(acc_list)
        print("curr_mean: ", curr_mean)
        f.write(json.dumps({"subject": sub, "acc": acc}) + "\n")
        f.flush()

      mean_acc = sum(acc_list)/len(acc_list)
      print("mean accuracy: ", mean_acc)
      f.write(json.dumps({"mean_acc": mean_acc}) + "\n")
      f.flush()

if __name__ == '__main__':
  app.run(main)


'''
large test size
large lr
more training
large batch
adam grads
dimred false
domai imbalance ?
10, 30 %, greatsdomain seems completely fucked at 10%
'''