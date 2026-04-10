from collections.abc import Iterable

import datasets
import jax.numpy as jnp
import numpy as np
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft.peft_trainer import TrainingInput  # pylint: disable=g-importing-member
from tunix.sft.eval.data_selection import get_training_dataset, get_validation_dataset
import tunix.sft.eval.chat_templates as chat_templates
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch
from tunix.sft.eval.colm import prompt_utils as colm_eval
import math
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


def get_splits(ds, split_ratio):
  split_index = int((1 - split_ratio) * len(ds))
  print("SPLIT", split_index)
  train_ds = ds.select(range(0, split_index))
  val_ds = ds.select(range(split_index, len(ds)))
  return train_ds, val_ds



def debug(dir_path):
  total = 0
  for p in sorted(Path(dir_path).rglob("*")):
    if p.is_file() and p.suffix.lower() in {".jsonl", ".jsonls", ".csv"}:
      n = sum(1 for _ in p.open("rb"))
      print(f"{n:>10}  {p.name}")
      total += n
  print("-" * 40)
  print(f"{total:>10}  TOTAL")



def filter_all_ignored_labels(ds, *, num_proc=10, desc=None, debug=False):
  def has_valid_label(example):
    # if debug:
    #   print(example["input_ids"])
    # labels is a torch.Tensor here
    return (example["labels"] != -100).any().item()
  
  total = len(ds)
  ds = ds.filter(
      has_valid_label,
      num_proc=num_proc,
      desc=desc or "Filtering samples with all labels = -100",
  )
  skipped = total - len(ds)
  print(f"Skipped {skipped}/{total} ({skipped / total:.2%}) samples")
  return ds


def make_jax_collate(tok, num_train_sources, max_target_length, subsel_bs=None, include_full=None, include_sourcemasks=False):
    collator = DataCollatorForSeq2Seq(
      tokenizer=tok,
      model=None,
      padding="max_length",
      max_length=max_target_length,
      label_pad_token_id=-100,
      return_tensors="np",
    )

    def jax_collate(features):
      ret = {}
      if "source" in features[0].keys():
        sources = [to_np_int32(ex["source"]) for ex in features]
        if include_sourcemasks:
          source_meta = get_source_masks(sources, subsel_bs, num_train_sources, include_full=include_full)
        else:
          source_meta = {"sources": jnp.array(sources, dtype=jnp.int32)}
        ret["meta"] = source_meta

      batch = collator(features)
      input_ids = jnp.asarray(batch["input_ids"], dtype=jnp.int32)
      labels = jnp.asarray(batch["labels"], dtype=jnp.int32)
      input_mask = (labels != -100).astype(jnp.int32)
      ret = {"input_tokens": input_ids, "input_mask": input_mask, **ret}
      return ret

    return jax_collate

def to_np_int32(x):
    if hasattr(x, "numpy"):  # tensor
        return x.cpu().numpy().astype(jnp.int32)
    return jnp.asarray(x, dtype=jnp.int32)  # numpy or list


class InfiniteLoader:
  def __init__(self, loader):
    self.loader = loader
    self.iterator = iter(loader)

  def __iter__(self):
    return self

  def __next__(self):
    try:
      return next(self.iterator)
    except StopIteration:
      self.iterator = iter(self.loader)  # new epoch
      return next(self.iterator)

  def __len__(self):
    # number of batches per epoch
    return len(self.loader)

# def make_seeded_loader(ds, batch_size, collate_fn, seed=42, shuffle=True, num_workers=0, infinite=False):
#   g = torch.Generator()
#   g.manual_seed(seed)
#   loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn,
#                       shuffle=shuffle, generator=g, num_workers=num_workers, drop_last=True)

#   if infinite: return InfiniteLoader(loader)
#   return loader

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

def make_seeded_loader(ds, batch_size, collate_fn, seed=42, shuffle=True,
                       num_workers=0, infinite=False,
                       domain_weights=None, src2id=None):
  g = torch.Generator().manual_seed(seed)

  sampler = None
  if domain_weights is not None and len(domain_weights) > 0:
    import json
    from tqdm import tqdm
    with open(domain_weights, "r") as f: weight_map = json.load(f)["train_domain_weights"]
    weight_map = {src2id[k]: weight_map[k] for k in weight_map.keys()}
    assert weight_map is not None
    assert src2id is not None
    print("Doing weighted sampling with", weight_map)
    weights = []
    for i in tqdm(range(len(ds)), desc="Mapping Weights"):
      src = ds[i]["source"].item()
      assert src is not None
      wt = weight_map[src]
      weights.append(wt)
    weights = torch.tensor(weights, dtype=torch.float)

    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=g)
    shuffle = False  # sampler and shuffle are mutually exclusive

  loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler,
                      shuffle=shuffle, generator=g, num_workers=num_workers, drop_last=True)

  return InfiniteLoader(loader) if infinite else loader


TIGER_DICT = {
    "instruction": "instruction",
    "response": "output",
    "source": "source",
    "ds_name": "tiger"
  }
METAMATH_DICT = {
    "instruction": "query",
    "response": "response",
    "source": "type",
    "ds_name": "metamath"
  }
MOL_DICT = {
    # "instruction": ["instruction", "input"],
    "instruction": "instruction",
    "input": "input",
    "response": "output",
    "source": "source",
    "ds_name": "MOL"
  }

def process_data(dataset, data_dict, verbose=True):
  """raw_sources: list[str] -> (data_sources: list[int], all_data_sources: list[str], num_sources: int)"""
  raw_sources = []
  
  key_instruction = data_dict["instruction"]
  key_input = data_dict.get("input", None)
  key_response = data_dict["response"]
  key_source = data_dict["source"]
  ds_name = data_dict["ds_name"]

  dataset = dataset.filter(lambda ex: bool(ex[key_response]))

  for ex in dataset:
    raw_sources.append(ex.get(key_source, 0))

  all_data_sources = sorted(set(raw_sources))
  src2id = {s: i for i, s in enumerate(all_data_sources)}
  if verbose:
    print("#"*50)
    print(src2id)
    print("#"*50)
  data_sources, num_sources = [src2id[s] for s in raw_sources], len(all_data_sources)

  # def get_input(ex):
  #   if isinstance(key_instruction, str):
  #     return ex[key_instruction]
  #   if isinstance(key_instruction, list):
  #     msg = "\n".join([ex[key] for key in key_instruction])
  #     return msg
  #   raise NotImplementedError

  def get_msg(ex):
    if key_input is None:
      msg = [
        {"role": "user", "content": ex[key_instruction]},
        {"role": "assistant", "content": ex[key_response]},
      ]
    else:
      msg = [
        {"role": "user", "content": ex[key_instruction]},
        {"role": "user", "content": ex[key_input]},
        {"role": "assistant", "content": ex[key_response]},
      ]

    return msg

  def proc(ex, idx):
    ret = {
      "dataset": ds_name,
      "id": f"{idx}",
      "messages": get_msg(ex),
      "source": data_sources[idx]
    }
    return ret
  dataset = dataset.map(
      proc,
      with_indices=True,
      # remove_columns=[key_instruction, key_response]
  )
  return dataset, src2id, num_sources

def get_eval_greats(tokenizer, config, max_target_length):
  ddir = "/home//temp/data"
  n_val = 2000
  task = "mmlu"
  # subject = get_subjects("/home//temp/data/eval/mmlu/dev")
  subject =  [
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
  print("Subjects", subject)
  wordy_dev = config["task_config"]["config"]["wordydev"]
  eval_ds = get_validation_dataset.get_dataset(
      task,
      data_dir=ddir,
      tokenizer=tokenizer,
      max_length=max_target_length,
      validation=True,
      k=n_val,
      subject=subject,
      append_choice_text="eval" in wordy_dev
  )
  eval_ds.set_format(type="pt")
  
  dev_ds = get_validation_dataset.get_dataset(
      task,
      data_dir=ddir,
      tokenizer=tokenizer,
      max_length=max_target_length,
      validation=True,
      k=n_val,
      subject=subject,
      append_choice_text="dev" in wordy_dev
  )
  dev_ds.set_format(type="pt")

  return eval_ds, dev_ds

def convert_to_chatformat(user, asst, idx):
  ret = {
    "dataset": "tiger",
    "id": f"{idx}",
    "messages": [
      {"role": "user", "content": user},
      {"role": "assistant", "content": asst},
    ],
  }
  return ret

from datasets import Dataset
def get_colm(tokenizer, shots, max_seq_length, mask_input, cands=[]):
  if not len(cands):
    # FIXME: we skip POT demonstrations for now, since 
    # code assisted eval is out of scope atm.
    cands = [k for k in  colm_eval.get_ex_dict("").keys() if "_pot" not in k]
  src2id = {k: id for id, k in enumerate(set(sorted(cands)))}
  dataset = {"input_ids": [], "attention_mask": [], "labels": [], "source": []}
  i = 0
  for ds in cands:
    exs = colm_eval.get_examples(ds, shots, "")
    for ex in exs:
      ques, resp = ex[0], ex[1]
      # ret, text = chat_templates.tokenize_chat_template(
      #   tokenizer, convert_to_chatformat(ques, resp, i), 
      #   max_seq_length, add_generation_prompt=False,
      #   return_text=True,
      # )
      ret, text = chat_templates.tokenize_prompt_alpaca(tokenizer, [], ques, 
        max_seq_length, resp=resp, mask_value=-100, return_text=True, mask_input=mask_input)
      ret = {k: v.flatten() for k,v in ret.items()}
      if i < 5:
        print(">>>val ques", ques)
        print(">>>val resp", resp)
        print(">>>val text", text)
        print(">>>Val ret", ret)
        print("############")
      i += 1

      dataset["source"].append(src2id[ds])
      for k, v in ret.items():
        dataset[k].append(v)
  dataset = Dataset.from_dict(dataset)

  # for i in range(5):
  #   print(dataset[i])
  #   print("##################")

  dataset.set_format(type="pt")
  return dataset


def split_by_group(ds, group_column, test_size=0.1, seed=42, shuffle=True):
    assert isinstance(ds, datasets.Dataset)
    unique_groups = set(ds[group_column])

    train_parts, test_parts = [], []

    for i, g in enumerate(unique_groups):
        subset = ds.filter(lambda x: x[group_column] == g)
        split = subset.train_test_split(
            test_size=test_size,
            seed=seed + i,
            shuffle=shuffle,
        )
        train_parts.append(split["train"])
        test_parts.append(split["test"])

    train_ds = datasets.concatenate_datasets(train_parts)
    test_ds = datasets.concatenate_datasets(test_parts)

    if shuffle:
        train_ds = train_ds.shuffle(seed=seed)
        test_ds = test_ds.shuffle(seed=seed + 1)

    return train_ds, test_ds

def repeat_dataset(ds, times, seed=42):
    out = []
    for i in range(times):
        out.append(ds.shuffle(seed + i))
    return datasets.concatenate_datasets(out)

def split_data(obj, group_column=None, test_size=0.1, seed=42, shuffle=True):
  if group_column is None:
    split = obj.train_test_split(test_size=test_size, seed=seed, shuffle=shuffle)
    return split["train"], split["test"]

  return split_by_group(obj, group_column, test_size, seed, shuffle)


def create_datasets(
  dataset_name: str,
  global_batch_size: int,
  eval_global_batch_size: int,
  max_target_length: int,
  num_train_epochs: int | None,
  tokenizer: tokenizer_lib.Tokenizer,
  *,
  split_ratio: float = 0.005,
  cache_dir = None,
  answer_only_mask: bool = True,
  subsel_bs=None,
  config = None
) -> tuple[Iterable[TrainingInput], Iterable[TrainingInput]]:

  if dataset_name in ["TIGER-Lab/MathInstruct", "meta-math/MetaMathQA", "zjunlp/Mol-Instructions"]:
    print(f"@@@@@@@@ Training on {dataset_name}")
    tok = tokenizer._tokenizer
    tok.chat_template = chat_templates.qwen2_5_template
    task_cfg = config["task_config"]["config"]
    include_full = None
    if task_cfg["subsel"]["minority_full"]: include_full = task_cfg["subsel"]["minority_classes"]
    domain_weights=None
    if task_cfg["domain_weights"]["enabled"]:
      domain_weights=task_cfg["domain_weights"]["weights_dir"]
    # raw = datasets.load_dataset(dataset_name, split="train")
    # raw = raw.shuffle(seed=42)
    if dataset_name == "TIGER-Lab/MathInstruct":
      raw = datasets.load_dataset(dataset_name, split="train")
      raw = raw.shuffle(seed=42)
      raw, src2id, num_train_sources = process_data(raw, TIGER_DICT)
      train_ds = raw
      train_on_input = False
      
    elif dataset_name == "meta-math/MetaMathQA":
      raw = datasets.load_dataset(dataset_name, split="train")
      raw = raw.shuffle(seed=42)
      raw = raw.train_test_split(test_size=10000, seed=42, shuffle=False)
      raw, _ = raw["train"], raw["test"]
      raw, src2id, num_train_sources = process_data(raw, METAMATH_DICT)
      train_ds = raw
      train_on_input = False
      

    elif dataset_name == "zjunlp/Mol-Instructions":
      cfgs = ['Molecule-oriented Instructions', 'Protein-oriented Instructions', 'Biomolecular Text Instructions']
      keys = [
        "description_guided_molecule_design",
        "forward_reaction_prediction",
        "reagent_prediction",
        "retrosynthesis"
      ]
      raw = datasets.load_dataset("zjunlp/Mol-Instructions", cfgs[0], trust_remote_code=True)
      # extract test set per group add source col and concat
      train_ds = datasets.concatenate_datasets([
        (split := raw[k].train_test_split(test_size=1000, seed=42, shuffle=True))['train']
            .add_column("source", [k] * len(split['train']))
        for k in keys
      ])
      train_ds = train_ds.shuffle(seed=42)
      train_ds, src2id, num_train_sources = process_data(train_ds, MOL_DICT)
      
      # sauce: https://github.com/zjunlp/Mol-Instructions/blob/main/demo/finetune.py#L49C9-L49C24
      train_on_input = True

    else:
      raise NotImplementedError

    eval_source = task_cfg["subsel"]["eval_source"]
    dev_source = task_cfg["subsel"]["dev_source"]
    print("EVAL SOURCE", eval_source)
    print("DEV SOURCE", dev_source)
    print("train on input", train_on_input)

    # TODO: split ratio things
    if eval_source == "train":
      print("Using random split from train as eval")
      train_ds, eval_ds = split_data(train_ds, group_column=None,
                                       test_size=split_ratio, seed=42, shuffle=True)
      eval_ds = get_training_dataset.encode_datav2(
          eval_ds, tok, max_target_length, mask_input=not train_on_input)

    elif eval_source == "colm":
      print("Using COLM as eval")
      eval_ds = get_colm(tok, 5, max_target_length, mask_input=not train_on_input)
    

    if dev_source == "train":
      print("Using random split from train as dev")
      train_ds, dev_ds = split_data(train_ds, group_column="source",
                                       test_size=split_ratio, seed=42, shuffle=True)
      dev_ds = get_training_dataset.encode_datav2(
          dev_ds, tok, max_target_length, mask_input=not train_on_input)

    elif dev_source == "eval":
      print("Using eval as dev")
      dev_ds = eval_ds

    elif dev_source == "colm":
      print("Using COLM as dev")
      dev_ds = get_colm(tok, 5, max_target_length, mask_input=not train_on_input)
    
    # train_ds = get_training_dataset.encode_data(
    #     raw, tok, max_target_length, func_name="encode_with_chat_template")
    # train_ds = repeat_dataset(train_ds, 5)
    train_ds = get_training_dataset.encode_datav2(
        train_ds, tok, max_target_length, mask_input=not train_on_input, verbose=True)
    
    # for i in range(5):
    #   print(train_ds[i])
    #   print("train_ds##################")
    # print("done 1")
    # subsel_bs = config['batch_size']
    jax_collate = make_jax_collate(tok, num_train_sources, max_target_length, subsel_bs=subsel_bs,
                                   include_full=include_full, include_sourcemasks=True)
    jax_collate_dev = make_jax_collate(tok, num_train_sources, max_target_length, subsel_bs=subsel_bs,
                                    include_sourcemasks=False)
    train_ds = filter_all_ignored_labels(train_ds, num_proc=10, desc="Filtering Train")
    train_ds = make_seeded_loader(train_ds, batch_size=global_batch_size, 
                                  domain_weights=domain_weights,
                                  collate_fn=jax_collate, seed=42, infinite=True, src2id=src2id)

    eval_ds = filter_all_ignored_labels(eval_ds, num_proc=10, desc="Filtering Val")
    eval_ds = make_seeded_loader(eval_ds, batch_size=eval_global_batch_size, 
                                  collate_fn=jax_collate_dev, seed=42, src2id=src2id)
    
    dev_ds = filter_all_ignored_labels(dev_ds, num_proc=10, desc="Filtering Dev")
    dev_ds = make_seeded_loader(dev_ds, batch_size=eval_global_batch_size, 
                                  collate_fn=jax_collate_dev, seed=42, src2id=src2id)

    return train_ds, eval_ds, dev_ds, {"num_train_sources": num_train_sources}
  elif dataset_name == "greats":
    data_dir = [
      "/home//temp/data/train/processed/cot/cot_data.jsonl",
      "/home//temp/data/train/processed/dolly/dolly_data.jsonl",
      "/home//temp/data/train/processed/flan_v2/flan_v2_data.jsonl",
      "/home//temp/data/train/processed/oasst1/oasst1_data.jsonl",
      # "/home//temp/data/train/processed/oasst1/oasst1_data_coding.jsonl",
    ]
    
    print("###########TRAIN#############")
    debug("/home//temp/data/train")
    print("###########EVAL#############")
    debug("/home//temp/data/eval/mmlu/dev")

    ddir = "/home//temp/data"
    tok = tokenizer._tokenizer
    tok.chat_template = chat_templates.qwen2_5_template
    jax_collate = make_jax_collate(tok, max_target_length)

    train_ds = get_training_dataset.get_training_dataset(data_dir, tok, max_target_length, seed=42)
    print("done 1")
    train_ds = filter_all_ignored_labels(train_ds, num_proc=10)
    train_ds = make_seeded_loader(train_ds, batch_size=global_batch_size, collate_fn=jax_collate, seed=42)

    n_val = 2000
    task = "mmlu"
    subject = get_subjects("/home//temp/data/eval/mmlu/dev")
    print("Subjects", subject)
    wordy_dev = config["task_config"]["config"]["wordydev"]
    eval_ds = get_validation_dataset.get_dataset(
        task,
        data_dir=ddir,
        tokenizer=tok,
        max_length=max_target_length,
        validation=True,
        k=n_val,
        subject=subject,
        append_choice_text="eval" in wordy_dev
    )
    eval_ds.set_format(type="pt")
    eval_ds = filter_all_ignored_labels(eval_ds, num_proc=10)
    eval_ds = make_seeded_loader(eval_ds, batch_size=eval_global_batch_size, collate_fn=jax_collate, seed=42)
    dev_ds = get_validation_dataset.get_dataset(
        task,
        data_dir=ddir,
        tokenizer=tok,
        max_length=max_target_length,
        validation=True,
        k=n_val,
        subject=subject,
        append_choice_text="dev" in wordy_dev
    )
    dev_ds.set_format(type="pt")
    dev_ds = filter_all_ignored_labels(dev_ds, num_proc=10)
    dev_ds = make_seeded_loader(dev_ds, batch_size=eval_global_batch_size, collate_fn=jax_collate, seed=42)

    return train_ds, eval_ds, dev_ds
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def scatter_active_to_full(active_vals, classes_active, MC, dtype=np.int32):
    """
    Convert a compact "active-class" vector into a full vector indexed by class id.
    """
    active_vals = np.asarray(active_vals, dtype=dtype)
    classes_active = np.asarray(classes_active, dtype=np.int32)

    assert classes_active.ndim == 1, "classes_active must be 1D."
    assert active_vals.ndim == 1, "active_vals must be 1D."
    assert active_vals.shape[0] == classes_active.shape[0], "active_vals and classes_active must align."
    assert int(MC) > 0, "MC must be positive."
    assert classes_active.min(initial=0) >= 0 and classes_active.max(initial=-1) < int(MC), (
        "classes_active must be in [0, MC)."
    )
    assert np.unique(classes_active).size == classes_active.size, "classes_active must be unique."

    full = np.zeros((int(MC),), dtype=dtype)
    full[classes_active] = active_vals
    return full

def _increase_array_to_threshold_masked(arr, threshold, eligible_mask):
    """
    Masked version of increase_array_to_threshold with identical behavior on the eligible subset.

    Key property:
      If eligible_mask is all True, this is exactly increase_array_to_threshold(arr, threshold).

    We do the same lexsort order, but only over eligible indices, and increment in that fixed order
    round-robin.
    """
    values = np.asarray(arr, dtype=np.int32).copy()
    eligible_mask = np.asarray(eligible_mask, dtype=bool)

    assert values.ndim == 1 and eligible_mask.ndim == 1
    assert values.shape[0] == eligible_mask.shape[0]
    assert np.all(values >= 0), "arr must be nonnegative."

    need = int(threshold) - int(values.sum())
    if need < 0:
        raise ValueError("threshold must be >= current sum")
    if need == 0:
        return values

    eligible_idx = np.where(eligible_mask)[0]
    if eligible_idx.size == 0:
        raise ValueError("No eligible indices to increase but threshold requires increases.")

    order_local = np.lexsort((eligible_idx, values[eligible_idx]))  # indices into eligible_idx
    order = eligible_idx[order_local]                               # actual indices in [0..len(values)-1]

    n = int(order.size)
    for step in range(need):
        i = order[step % n]
        values[i] += 1
    return values


def _validate_inputs_and_active_view(
    src, N, MC, B,
    uniq, classes_active, y, freqs_active,
    include_full,
    strategy, per_class_start,
):
    """
    Validates inputs + active-class view invariants + include_full feasibility.

    Returns:
      include_full_arr: [K] int32
      forced_active: [C_active] bool
      B_forced: int
      B_free: int
      N_free: int
    """
    assert src.ndim == 1 and src.shape[0] == N
    assert N > 0
    assert MC > 0
    assert 0 <= B <= N
    assert src.min() >= 0 and src.max() < MC

    C_active = int(classes_active.size)
    assert uniq.ndim == 1 and classes_active.ndim == 1
    assert y.ndim == 1 and y.shape == (N,)
    assert C_active >= 1
    assert np.all(uniq[:-1] < uniq[1:]), "uniq must be strictly increasing."
    assert np.all(classes_active[:-1] < classes_active[1:]), "classes_active must be strictly increasing."
    assert y.min() >= 0 and y.max() < C_active, "inv/y must be in [0, C_active)."
    assert np.array_equal(uniq[y].astype(src.dtype), src), "Reconstruction uniq[inv] != src."
    assert np.array_equal(classes_active[y].astype(src.dtype), src), "Reconstruction classes_active[y] != src."

    assert freqs_active.shape == (C_active,)
    assert int(freqs_active.sum()) == N, "Active class counts must sum to N."
    assert np.all(freqs_active > 0), "All active classes must have positive frequency."

    if strategy == "proportional":
        assert per_class_start in ("floor", "ceil"), "per_class_start must be 'floor' or 'ceil'."
    elif strategy == "none":
        pass
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    if include_full is None:
        include_full_arr = np.empty((0,), dtype=np.int32)
    else:
        include_full_arr = np.asarray(include_full, dtype=np.int32).reshape(-1)

    if include_full_arr.size > 0:
        assert include_full_arr.min() >= 0 and include_full_arr.max() < MC, "include_full must be in [0, MC)."
        assert np.unique(include_full_arr).size == include_full_arr.size, "include_full must be unique."

    forced_active = np.isin(classes_active, include_full_arr)
    B_forced = int(freqs_active[forced_active].sum())
    B_free = int(B - B_forced)
    assert B_free >= 0, f"include_full forces selecting {B_forced} items, which exceeds B={B}."

    N_free = int(freqs_active[~forced_active].sum())
    assert B_free <= N_free, f"Remaining budget B_free={B_free} exceeds remaining available N_free={N_free}."

    return include_full_arr, forced_active, B_forced, B_free, N_free


def _validate_packed_outputs(
    src, MC, B,
    classes_active, k_active,
    k_per_class_vec, out_offsets_vec,
):
    """
    Post-allocation logic checks: scatter correctness, availability, offsets correctness.
    """
    src = np.asarray(src, dtype=np.int32)
    classes_active = np.asarray(classes_active, dtype=np.int32)
    k_active = np.asarray(k_active, dtype=np.int32)
    k_per_class_vec = np.asarray(k_per_class_vec, dtype=np.int32)
    out_offsets_vec = np.asarray(out_offsets_vec, dtype=np.int32)

    C_active = int(classes_active.size)

    assert k_active.shape == (C_active,)
    assert k_per_class_vec.shape == (MC,)
    assert out_offsets_vec.shape == (MC,)

    assert np.all(k_active >= 0)
    assert np.all(k_per_class_vec >= 0)
    assert int(k_active.sum()) == int(B)
    assert int(k_per_class_vec.sum()) == int(B)

    assert np.array_equal(k_per_class_vec[classes_active], k_active), "Scatter mismatch: full[classes_active] != k_active."
    non_active = np.ones(MC, dtype=bool)
    non_active[classes_active] = False
    assert np.all(k_per_class_vec[non_active] == 0), "Non-active classes must have zero allocation."

    counts_by_class = np.bincount(src, minlength=MC).astype(np.int32)
    excess = k_per_class_vec - counts_by_class
    assert np.all(excess <= 0), (
        "Requested more than available in some real classes. "
        f"Max excess={int(excess.max())}."
    )

    prefix = np.zeros((MC,), dtype=np.int32)
    if MC > 1:
        prefix[1:] = np.cumsum(k_per_class_vec[:-1], dtype=np.int64).astype(np.int32)
    assert np.array_equal(out_offsets_vec, prefix), "out_offsets_vec must be prefix-sum of k_per_class_vec."

    assert np.all(out_offsets_vec >= 0) and np.all(out_offsets_vec <= B), "Offsets out of range."
    assert np.all(out_offsets_vec[:-1] <= out_offsets_vec[1:]), "Offsets must be non-decreasing."
    assert np.all(out_offsets_vec + k_per_class_vec <= B), "Class slice overruns global cap B."
    if MC > 0:
        assert int(out_offsets_vec[-1] + k_per_class_vec[-1]) == int(B), "Last class slice must end at B."


def get_source_masks(
    sources,            # [N] class ids; guaranteed to be in [0..MC-1]
    B: int,             # total number of items to select across all classes
    max_classes: int,   # MC; fixed universe of class ids 0..MC-1
    strategy: str = "proportional",
    per_class_start: str = "floor",
    include_full=None,  # list/array of real class ids to fully include (ignored for strategy="none")
):
    """
    Build per-class selection metadata.

    include_full (only meaningful for strategy="proportional"):
      Real class ids (0..MC-1). For these classes, we force selecting all examples
      from that class (k[c] = count_in_data[c]). Remaining budget is allocated to
      other classes according to `strategy`.
    """
    src = np.asarray(sources, dtype=np.int32)
    N = int(src.shape[0])
    MC = int(max_classes)
    B = int(B)

    if src.ndim != 1:
        raise ValueError(f"sources must be 1D, got shape={src.shape}.")
    if N <= 0:
        raise ValueError("sources must be non-empty.")
    if MC <= 0:
        raise ValueError("max_classes must be positive.")
    if B < 0 or B > N:
        raise ValueError(f"B must be in [0, N]. Got B={B}, N={N}.")
    if src.min() < 0 or src.max() >= MC:
        raise ValueError(f"sources must be in [0, {MC-1}] (got min={src.min()}, max={src.max()}).")

    uniq, inv = np.unique(src, return_inverse=True)
    classes_active = uniq.astype(np.int32)
    y = inv.astype(np.int32)
    C_active = int(classes_active.size)

    if C_active > MC:
        raise ValueError(f"max_classes={MC} < actual active classes={C_active}. Increase max_classes.")

    freqs_active = np.bincount(y, minlength=C_active).astype(np.int32)

    include_full_eff = None if strategy == "none" else include_full

    include_full_arr, forced_active, B_forced, B_free, N_free = _validate_inputs_and_active_view(
        src=src, N=N, MC=MC, B=B,
        uniq=uniq, classes_active=classes_active, y=y, freqs_active=freqs_active,
        include_full=include_full_eff,
        strategy=strategy, per_class_start=per_class_start,
    )

    if strategy == "none":
        num_per_class_active = np.int32([B])
        classes_active = np.array([0], dtype=np.int32)
        C_active = 1
        freqs_active = np.array([N], dtype=np.int32)
        y = np.zeros(N, dtype=np.int32)

        assert num_per_class_active.shape == (1,)
        assert int(num_per_class_active.sum()) == B

        k_active = np.minimum(num_per_class_active, freqs_active).astype(np.int32)
        assert int(k_active.sum()) == B

    elif strategy == "proportional":
        # Forced: take all available from forced classes (active space)
        k_forced_active = np.where(forced_active, freqs_active, 0).astype(np.int32)

        # Free proportional allocation over non-forced classes, total B_free
        raw_free = np.zeros_like(freqs_active, dtype=np.float64)
        if B_free > 0:
            denom = float(freqs_active[~forced_active].sum())
            assert denom > 0.0
            raw_free = (freqs_active.astype(np.float64) / denom) * float(B_free)
            raw_free = raw_free * (~forced_active).astype(np.float64)

        if per_class_start == "floor":
            num_free_active = np.floor(raw_free).astype(np.int32)
        else:
            num_free_active = np.ceil(raw_free).astype(np.int32)

        num_free_active = _increase_array_to_threshold_masked(
            num_free_active, B_free, eligible_mask=(~forced_active)
        )

        if B_free == 0:
            assert int(num_free_active.sum()) == 0, "B_free=0 but free allocation is nonzero."

        num_per_class_active = (k_forced_active + num_free_active).astype(np.int32)

        assert num_per_class_active.shape == (C_active,), "Allocation must be length C_active."
        assert np.issubdtype(num_per_class_active.dtype, np.integer)
        assert np.all(num_per_class_active >= 0), "Allocation must be nonnegative."
        assert int(num_per_class_active.sum()) == B, "Allocation must sum exactly to B."

        over = num_per_class_active - freqs_active
        assert np.all(over <= 0), (
            "No-redistribution mode violated: allocated more than available in some active classes. "
            f"Max overallocation={int(over.max())}."
        )

        k_active = np.minimum(num_per_class_active, freqs_active).astype(np.int32)

        assert int(k_active.sum()) == B, (
            "Availability capping reduced the total below B. "
            "Either adjust allocation to respect class counts or add redistribution."
        )
        assert np.all(k_active >= 0), "k_active must be nonnegative."
        assert k_active.shape[0] == classes_active.shape[0], "k_active must align with classes_active."

        if include_full_arr.size > 0:
            assert np.all(k_active[forced_active] == freqs_active[forced_active]), (
                "include_full classes must take all available examples."
            )

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    k_per_class_vec = scatter_active_to_full(k_active, classes_active, MC, dtype=np.int32)

    out_offsets_vec = np.zeros((MC,), dtype=np.int32)
    if MC > 1:
        out_offsets_vec[1:] = np.cumsum(k_per_class_vec[:-1], dtype=np.int64).astype(np.int32)

    _validate_packed_outputs(
        src=src, MC=MC, B=B,
        classes_active=classes_active, k_active=k_active,
        k_per_class_vec=k_per_class_vec, out_offsets_vec=out_offsets_vec,
    )

    k_per_class = np.repeat(k_per_class_vec[None, :], N, axis=0)
    out_offsets = np.repeat(out_offsets_vec[None, :], N, axis=0)

    meta = dict(
        k_eff=k_per_class,        # [N, MC]
        out_offsets=out_offsets,  # [N, MC]
        sources=src,
    )
    return meta
