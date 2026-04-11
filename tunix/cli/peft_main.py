"""Main entry point for PEFT training."""
from collections.abc import Callable
from typing import Any
from absl import app
from flax import nnx
import jax
from tunix.cli import config
from tunix.cli import optax_ext
from tunix.cli.utils import model as model_lib
from tunix.cli.loss import *
from tunix.examples.data import retrieval_dataset as data_lib_retr
from tunix.examples.data import nomic_jax_dataset as data_lib_nomicjax
from tunix.examples.data import ift_dataset as data_lib_ift
from tunix.examples.data import nomic_dataset as data_lib_nomic
from tunix.sft import subset_trainer
from tunix.sft import utils
import optax
from typing import Any, Callable
from flax import nnx
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from functools import partial
from tunix.cli.beir import *
from tunix.cli.eval import *
from tunix.cli.loss import *
from flax.core import FrozenDict



def build_optimizer(cfg):
    # --- Scheduler ---
    '''
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    '''

    def linear_warmup_decay_schedule(warmup_steps, total_steps, peak_lr, end_lr=0.0):
      warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
      decay = optax.linear_schedule(peak_lr, end_lr, total_steps - warmup_steps)
      return optax.join_schedules([warmup, decay], [warmup_steps])

    # lr = 2e-5
    lr = cfg["optimizer_config"]["learning_rate"]
    warmup_ratio = cfg["optimizer_config"]["warmup_ratio"]
    print("$$$$$$$$$$$$$$$$$$$")
    print("LEARNING RATE", lr)
    print("$$$$$$$$$$$$$$$$$$$")

    total_steps = cfg["training_config"]["max_steps"]

    warmup_steps = int(warmup_ratio*total_steps)
    # warmup_steps = 700

    decay_steps = total_steps - warmup_steps
    weight_decay = 0.0
    grad_norm = 1.0

    # schedule = optax.warmup_cosine=_decay_schedule(
    #   init_value=0.0,
    #   peak_value=lr,
    #   warmup_steps=warmup_steps,
    #   decay_steps=decay_steps,
    #   end_value=0.0,
    # )

    schedule = linear_warmup_decay_schedule(
      warmup_steps=warmup_steps,
      total_steps=total_steps,
      peak_lr=lr,
      end_lr=0.0
    )
    optim = cfg["optimizer_config"]["opt_type"]
    print("$$$$$$$$$$$$$$$$$$ OPTIM", optim)
    
    if optim == "sgd":
      tx = optax.sgd(learning_rate=schedule)
    if optim == "muon":
      tx = optax.contrib.muon(learning_rate=schedule, weight_decay=weight_decay)
    if optim == "asgo":
      tx = optax_ext.asgo(learning_rate=schedule, weight_decay=weight_decay)
    if optim == "dasgo":
      tx = optax_ext.dasgo(learning_rate=schedule, weight_decay=weight_decay)
    if optim == "shampoo":
      tx = optax_ext.shampoo(learning_rate=schedule, weight_decay=weight_decay)
    if optim == "adamw":
      # tx = optax.schedules.inject_hyperparams(optax.adamw)(learning_rate=schedule, weight_decay=weight_decay)
      tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

    if optim not in {"sgd", "muon", "asgo", "dasgo", "shampoo", "adamw"}:
      raise ValueError(f"Unknown optimizer type: {optim}")

    tx = optax.chain(
      optax.clip_by_global_norm(grad_norm),
      tx,
    )

    return tx, schedule

def build_optimizerhead(cfg):
    def linear_warmup_decay_schedule(warmup_steps, total_steps, peak_lr, end_lr=0.0):
      warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
      decay = optax.linear_schedule(peak_lr, end_lr, total_steps - warmup_steps)
      return optax.join_schedules([warmup, decay], [warmup_steps])

    lr = cfg["task_config"]["config"]["gradsapprox"]["lr_head"]
    warmup_ratio = cfg["optimizer_config"]["warmup_ratio"]
    print("$$$$$$$$$$$$$$$$$$$")
    print("LEARNING RATE", lr)
    print("$$$$$$$$$$$$$$$$$$$")

    total_steps = cfg["training_config"]["max_steps"]

    warmup_steps = int(warmup_ratio*total_steps)
    # warmup_steps = 700

    decay_steps = total_steps - warmup_steps
    weight_decay = 0.0
    grad_norm = 1.0

    schedule = linear_warmup_decay_schedule(
      warmup_steps=warmup_steps,
      total_steps=total_steps,
      peak_lr=lr,
      end_lr=0.0
    )
    
    tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

    tx = optax.chain(
      optax.clip_by_global_norm(grad_norm),
      tx,
    )

    return tx, schedule



def fix_pad_mask(tokens, pad_id):
    # assume padding only at end and pad_id == eos_id
    T = tokens.shape[1]
    is_pad = tokens == pad_id

    # count trailing pads
    rev = jnp.flip(is_pad, 1)
    num_trailing = jnp.cumsum(rev, 1)[:, -1]

    # keep 1 of them as EOS
    pad_count = jnp.clip(num_trailing - 1, 0, T)

    # build final mask (True = real or EOS, False = pad)
    idx = jnp.arange(T)
    return idx < (T - pad_count[:, None])

def prepare_input(tokens, pad_id, is_causal=True, is_retrieval=False):
  if is_retrieval:
    pad_mask = fix_pad_mask(tokens, pad_id)
  else:
    # pad_mask = tokens != pad_id
    pad_mask = jax.numpy.ones_like(tokens)

  positions = utils.build_positions_from_mask(pad_mask)
  if is_causal:
    attention_mask = utils.make_causal_attn_mask(pad_mask)
  else:
    attention_mask = utils.make_self_attn_mask(pad_mask)

  return {
    "input_tokens": tokens,
    "positions": positions,
    "attention_mask": attention_mask,
  }

def gen_model_input_fn_ift(x: subset_trainer.TrainingInput, pad_id, is_causal):
  ret = prepare_input(x["input_tokens"], pad_id, is_causal)
  ret["input_mask"] = x["input_mask"]
  if "meta" in x.keys():
    ret["meta"] = x["meta"]
  return ret

def gen_model_input_fn_retr(x: subset_trainer.TrainingInput, pad_id, is_causal, is_retrieval):
  # ret = {k: prepare_input(x[k], pad_id, is_causal, is_retrieval) for k in x}
  return x

def alpha_exponential(step, alpha0, alpha_min, k):
  step = step.astype(jnp.float32)
  return alpha_min + alpha0 * jnp.exp(-k * step)

def linear_schedule(step, start, end, total_steps):
  step = step + 1
  fraction = jnp.clip(step / total_steps, 0.0, 1.0)
  return start + fraction * (end - start)

def ramp_schedule(step, start, end, breakoff, total_steps):
  if (start == 0 and end == 0): return 0
  if (breakoff == -1): breakoff = total_steps
  step = step + 1
  fraction = jnp.clip(step / breakoff, 0.0, 1.0)
  return start + fraction * (end - start)

class PeftPipeline(config.HyperParameters):

  def run_peft_trainer(self):
    """Run the PEFT trainer."""
    mesh: jax.sharding.Mesh = self.create_mesh('model_config')
    model: nnx.Module | None = None
    tokenizer: Any | None = None
    my_gen_model_input_fn: (
        Callable[[subset_trainer.TrainingInput], dict[str, Any]] | None
    ) = None
    from omegaconf import OmegaConf
    self.config["task_config"]["config"] = OmegaConf.load(self.config["task_config"]["config"])
    print("Loaded task config", self.config["task_config"]["config"])
    print("##################################################################")
    print("##################################################################")
    # print("###########")
    # print("###########")
    
    print("----------------------------------------------------------")
    print(f"Subset Select Mode : {self.config['subset_select']['mode']}")
    print("----------------------------------------------------------")
    
    model, tokenizer_path = model_lib.create_model(
        self.config['model_config'], self.config['tokenizer_config'], mesh
    )

    if model is None:
      raise ValueError('model is None')
    tokenizer = model_lib.create_tokenizer(
        self.config['tokenizer_config'], tokenizer_path
    )
    
    steps = self.config["training_config"]["max_steps"]
    

    print("EOS", tokenizer.eos_id())
    print("PAD", tokenizer.pad_id())
    # assert tokenizer.pad_id() == tokenizer.eos_id()


    optimizer, schedule = build_optimizer(self.config)
    optimizer_last, schedule1 = build_optimizer(self.config)
    optimizer_head, schedule_head = build_optimizerhead(self.config)
    optimizer_head2, schedule_head = build_optimizerhead(self.config)
    
    
    task_config = self.config["task_config"]["config"]
    print("task config", task_config)
    
    config = {
      "task": self.config["task_config"]["task"],
      "qry_prefix": "", 
      "doc_prefix": "",
      "pad_id": tokenizer.pad_id(),
      "is_causal": True,
      "emb_mode": "eos",
      "sim_fn": "cosine",
      "train": {
        "remat_bs": self.config['remat_bs'],
      },
    }
    print("config", config)

    if config["task"] == "retrieval":
      assert tokenizer.pad_id() == tokenizer.eos_id()
      my_gen_model_input_fn = partial(gen_model_input_fn_retr, pad_id=config["pad_id"], 
                                        is_causal=config["is_causal"], is_retrieval=True)
      my_datalib = data_lib_nomicjax
      config["loss"] = {
        "mode": task_config["mode"],
        "lambda": lambda step: ramp_schedule(step, task_config["lamb"][0],task_config["lamb"][1],task_config["lamb"][2], steps),
        # "lambda": lambda step: ramp_schedule(step, 1.0, 1e-2, 512),
        # "lambda": lambda step: linear_schedule(step, 50.0, 1e-4, steps)
      }
      config["qry_prefix"] = "search_query: " # Space at end
      config["doc_prefix"] = "search_document: "

      dsspec = self.config["dataset_name"]
      cache_dir = self.config["cache_dir"]
      dataset_name=dsspec
      config["eval_sources"] = (
        # "NanoTouche2020Retrieval", "NanoClimateFeverRetrieval", 
        # "MSMARCO",
        "TRECCOVID", "SpartQA", "HagridRetrieval",
        "SCIDOCS", "SciFact", "ArguAna", "NFCorpus", "WinoGrande", "TempReasonL1", 
        # "WikipediaRetrievalMultilingual"
      )
      config = FrozenDict(config)
      def t(model, inputs, aux):
        return partial(loss_retr, config)(model, inputs, aux)
      train_loss = t
      eval_loss = t
      eval_fn = partial(eval_ir, config, tokenizer, prepare_input)

    if config["task"] == "ift":
      my_gen_model_input_fn = partial(gen_model_input_fn_ift, pad_id=config["pad_id"], is_causal=config["is_causal"])
      
      my_datalib = data_lib_ift
      train_loss = loss_unpacked
      eval_loss = default_loss_fn
      dataset_name=self.config['dataset_name']
      eval_fn = None
      cache_dir=None

    subsel = self.config["subset_select"]["enabled"]
    buffer = 1
    if subsel:
      buffer = self.config["subset_select"]["buffer"]
    
    train_ds, eval_ds, dev_ds, data_meta = my_datalib.create_datasets(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        global_batch_size=self.config['batch_size']*buffer,
        eval_global_batch_size=self.config['eval_batch_size'],
        max_target_length=self.config['max_target_length'],
        num_train_epochs=100,
        tokenizer=tokenizer,
        split_ratio=self.config['eval_split'],
        config=self.config,
        subsel_bs = self.config["batch_size"]*get_train(self.config)
    )
    # ds = train_ds._data_source
    ds = train_ds
    def get_len(ds):
      try: l = len(ds._data_source)
      except: 
        try: l = len(ds.dataset)
        except: l = len(ds)
      return l
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("Len Dataset(Million):", get_len(ds)/1e6)
    print("Num Batches:", get_len(ds)//self.config['batch_size'])
    print("Len Eval Dataset:", get_len(eval_ds))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # Optimizer is now created inside PeftTrainer based on _lora_enabled flag
    # if utils.is_lora_enabled(model):
    #     tx = optax.MultiSteps(optimizer, 1)
    #     tx = nnx.Optimizer(model, tx, wrt=nnx.LoRAParam)
    # else: skip — full FT optimizer is created inside PeftTrainer
    # jax.debug.print("opt {}", tx.opt_state.inner_opt_state[1].hyperparams)

    
    trainer = subset_trainer.PeftTrainer(
        model,
        optimizer,
        subset_trainer.TrainingConfig(
            **self.obtain_training_config_dict('training_config')
        ),
        optimizer_head=optimizer_head,
        optimizer_last=optimizer_last,
        optimizer_head2=optimizer_head2,
        has_aux=True,
        schedule=schedule,
        config=FrozenDict(self.config),
        train_loss=train_loss,
        eval_loss=eval_loss,
        eval_fn=eval_fn
    )
    
    trainer = trainer.with_gen_model_input_fn(my_gen_model_input_fn)

    with mesh:
      trainer.train(train_ds, data_meta["num_train_sources"], eval_ds, dev_ds=dev_ds)

def get_train(config):
  if not config["subset_select"]["enabled"]: return 1
  _ratio = config["subset_select"]["ratio"]
  if config["subset_select"]["mode"] == "full": _ratio = 1
  _buffer = config["subset_select"]["buffer"]
  batch_to_buffer =int(_buffer*_ratio) 
  return batch_to_buffer


def main(argv, **kwargs):
  pipeline = PeftPipeline(argv, **kwargs)
  pipeline.run_peft_trainer()


if __name__ == '__main__':
  app.run(main)


'''
full eval and mteb
kink in the graph
eos is correct ?
gmv done
distillation 
nerf

full eval
seeds
redo std
variance exps


promot lambp == -100
prompt
mmlu eval
full model grad

check empty seq labels
lr -> 3e-4
small batch but grad accum

not mmlu
lr
smaller model

'''