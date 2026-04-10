# Optimizer Selection for Gradient Coresets

Training and evaluation utilities for SFT/PEFT, subset selection (including GREATS), and COLM-style math evaluation.

## 1) Environment Setup

### System assumptions
- Linux
- Python 3.10+
- CUDA/JAX setup compatible with your GPU

### Create and activate Conda environment
```bash
conda create -n myenv python=3.10.12
conda activate myenv
```

### Install dependencies
From the repository root, run:
```bash
python install_packages.py
```

This will install all required dependencies (jax, flax, optax, transformers, etc.).

If you plan to use model download/tokenizers from Hugging Face:
```bash
export HF_TOKEN=<your_hf_token>
```

## 2) Project Structure

```
tunix/
│
├── cli/                          ← Training entry points and config
│   ├── base_config.yaml          ← Main experiment config (all hyperparameters)
│   ├── task_config_baseline.yaml ← Task-specific settings (IFT, retrieval)
│   ├── peft_main.py              ← Main training launcher
│   ├── generate.py               ← Inference / COLM evaluation launcher
│   ├── grpo_main.py              ← GRPO RL training launcher (Not Used Anymore)
│   ├── config.py                 ← HyperParameters config loader class
│   ├── loss.py                   ← Loss functions (IFT, retrieval, contrastive)
│   ├── optax_ext.py              ← Custom optimizers (ASGO, DASGO, Shampoo)
│   ├── eval.py                   ← IR / MTEB evaluation utilities
│   ├── beir.py                   ← BEIR benchmark helpers
│   └── utils/                    ← Config and training utilities
│
├── generate/                     ← Inference engine
│   ├── sampler.py                ← Main auto-regressive sampler
│   ├── beam_search.py            ← Beam search implementation
│   ├── base_sampler.py           ← Abstract sampler base class
│   ├── vllm_sampler.py           ← vLLM-backed sampler
│   ├── tokenizer_adapter.py      ← Unified tokenizer wrapper
│   ├── utils.py                  ← Attention masks, padding helpers
│   └── mappings.py               ← Weight name mapping utilities
│
├── models/                       ← Model architecture definitions
│   ├── gemma/                    ← Gemma/Gemma2 model
│   ├── gemma3/                   ← Gemma3 model
│   ├── llama3/                   ← Llama3 model
│   ├── qwen2/                    ← Qwen2.5 model
│   ├── qwen3/                    ← Qwen3 model (with MoE)
│   └── safetensors_loader.py     ← Shared concurrent safetensors loader
│
├── sft/                          ← Training infrastructure
│   ├── subset_trainer.py         ← Custom trainer with subset selection loop
│   ├── subsel_utils.py           ← All selection algorithms (GREATS, FacLoc etc.)
│   ├── peft_trainer.py           ← Base PEFT trainer
│   ├── checkpoint_manager.py     ← Orbax checkpoint save/restore
│   ├── metrics_logger.py         ← TensorBoard + W&B logging
│   ├── utils.py                  ← Attention masks, HBM monitoring
│   ├── hooks.py                  ← Training/data hook interfaces
│   ├── sharding_utils.py         ← JAX sharding helpers
│   ├── progress_bar.py           ← tqdm progress bar
│   ├── profiler.py               ← JAX profiler wrapper
│   ├── inflight_throttler.py     ← Async compute throttling
│   ├── system_metrics_calculator.py ← TFLOP measurement
│   └── eval/                     ← Evaluation utilities
│       ├── colm/                 ← COLM benchmark runner
│       ├── data_selection/       ← Dataset loaders
│       ├── mol/                  ← Molecule translation metrics
│       ├── mmlu_eval.py          ← Standalone MMLU evaluator
│       ├── tydiqa_eval.py        ← Standalone TydiQA evaluator
│       └── chat_templates.py     ← Prompt formatting templates
│
├── examples/
│   └── data/                     ← Data pipelines
│       ├── ift_dataset.py        ← IFT data pipeline
│       ├── retrieval_dataset.py  ← Retrieval/embedding data pipeline
│       ├── nomic_jax_dataset.py  ← Nomic embedding dataset
│       └── pretrain_dataset.py   ← Pre-training data pipeline
│
├── utils/                        ← Utility modules
│   ├── compat.py                 ← JAX/Flax version compatibility shims
│   └── container.py              ← ModuleList compatibility shim
│
├── oss/                          ← External resource utilities
│   └── utils.py                  ← GCS, Kaggle, HuggingFace download helpers
│
└── __init__.py                   ← Public Tunix API exports
```

## 3) Repo Notes Before Running

- Main SFT/PEFT training entrypoint: `cli/peft_main.py`
- Main generation/eval entrypoint (currently calls COLM eval by default): `cli/generate.py`
- Base config: `cli/base_config.yaml`

Important:
- `task_config.config` in `cli/base_config.yaml` must point to a valid YAML file. See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for detailed parameter documentation.
- `cli/peft_main.py` loads this path with OmegaConf at runtime.

## 4) LoRA Configuration

The training script supports both LoRA (Low-Rank Adaptation) fine-tuning and full model fine-tuning.

### Enable LoRA
To use LoRA fine-tuning, set the following in `cli/base_config.yaml`:
```yaml
lora_enabled: true
```

### Disable LoRA (Full Fine-tuning)
For full model fine-tuning, set:
```yaml
lora_enabled: false
```

## 5) Running the Main Training Script

From the repository root, run:
```bash
python -m tunix.cli.peft_main base_config.yaml > training.log 2>&1
```

This will:
1. Load configuration from `cli/base_config.yaml`
2. Set up the model and tokenizer
3. Initialize the selected optimizer and subset selection mode
4. Start training and save logs to `training.log`

## 6) Configuring Optimizer and Subset Selection

To change the optimizer and subset selection mode without modifying command-line arguments, edit `cli/base_config.yaml` directly.

### Available Optimizers
Edit `optimizer_config.opt_type` in `cli/base_config.yaml`:
- `adamw`
- `sgd`
- `muon`
- `asgo`
- `dasgo`
- `shampoo`

### Available Subset Selection Modes
Edit `subset_select.mode` and other subset selection parameters in `cli/base_config.yaml`:
- `mode`: `full`, `random`, `gradnorm`, `facloc`, `greats`, `joint`
- `ratio`: fraction of data to use (0.0 to 1.0)
- `buffer`: buffer size for subset selection

### Example Base Config Setup
```yaml
lora_enabled: true
optimizer_config:
  opt_type: asgo
  learning_rate: 1e-5
  warmup_ratio: 0.1
subset_select:
  enabled: true
  mode: greats
  ratio: 0.5
  buffer: 8
```

Then run:
```bash
python -m tunix.cli.peft_main base_config.yaml > training.log 2>&1
```

## 7) Configuration Quick Guide

In `cli/base_config.yaml`, key fields you will usually touch:

- `optimizer_config.opt_type`
  - supported here: `adamw`, `sgd`, `muon`, `asgo`, `dasgo`, `shampoo`
- `subset_select.enabled`
- `subset_select.mode`
  - available modes in subset selection path: `full`, `random`, `gradnorm`, `facloc`, `greats`, `joint`
- `subset_select.ratio`
- `subset_select.buffer`
- `task_config.config`
- `training_config.max_steps`
- `batch_size`
- `lora_enabled` (set to `true` for LoRA, `false` for full fine-tuning)

## 8) Run Training

All commands below are run from repository root.

### Base command
```bash
python -m tunix.cli.peft_main cli/base_config.yaml
```

### A) Baseline runs

#### Full-data baseline
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:full,ratio:1.0,buffer:1}
```

#### Random subset baseline
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:random,ratio:0.5,buffer:8}
```

#### GradNorm baseline
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:gradnorm,ratio:0.5,buffer:8}
```

#### Facility-location baseline
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:facloc,ratio:0.5,buffer:8}
```

### B) GREATS run
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:greats,ratio:0.5,buffer:8}
```

### C) Joint mode
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:joint,ratio:0.5,buffer:8}
```

## 9) Choose Optimizer

Set from config or CLI override via `optimizer_config.opt_type`.

### AdamW
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  optimizer_config={opt_type:adamw,learning_rate:1e-5,warmup_ratio:0.1}
```

### ASGO
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  optimizer_config={opt_type:asgo,learning_rate:1e-5,warmup_ratio:0.1}
```

### DASGO
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  optimizer_config={opt_type:dasgo,learning_rate:1e-5,warmup_ratio:0.1}
```

### Shampoo
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  optimizer_config={opt_type:shampoo,learning_rate:1e-5,warmup_ratio:0.1}
```

## 10) COLM Evaluation

Current `cli/generate.py` default behavior is to run `COLM_eval(...)` from `main`.

Run:
```bash
python -m tunix.cli.generate cli/base_config.yaml
```

What this does (current code behavior):
- Loads model + tokenizer
- Restores LoRA checkpoint from `training_config.checkpoint_root_directory` and `inference_restore_step`
- Runs evaluation on COLM datasets (`numglue`, `mmlu_mathematics`, `gsm8k`, `svamp`, `simuleq`, `deepmind`, `aqua`, `sat`)
- Writes per-dataset outputs and summary CSV under the result directory

Paths hardcoded in current COLM flow:
- dataset root: `/home/temp/CoLM/math_eval/dataset`
- output root pattern: `/home/temp/tunix/examples/sft/mtnt/results/<exp_name>/<ckpt_num>`

Ensure these paths exist or update them in `cli/generate.py` before running.

## 11) Common Issues and Fixes

### Missing HF token
Symptom: tokenizer/model load fails.
Fix:
```bash
export HF_TOKEN=<your_hf_token>
```

### task_config.config is empty
Symptom: OmegaConf load error in training startup.
Fix:
- Set `task_config.config` in `cli/base_config.yaml` to a real YAML file path.

### Unknown optimizer type
Symptom: `Unknown optimizer type: ...`
Fix:
- Use one of: `adamw`, `sgd`, `muon`, `asgo`, `dasgo`, `shampoo`.

### COLM path errors
Symptom: dataset not found or output path issues.
Fix:
- Create/adjust:
  - `/home/temp/CoLM/math_eval/dataset`
  - `/home/temp/tunix/examples/sft/mtnt/results/...`

## 12) Minimal Repro Commands

---

## 13) Configuration Parameter Reference

For detailed documentation of all configuration parameters across `base_config.yaml` and `task_config.yaml`, see [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

This reference includes:
- `model_config` block: Model loading, LoRA settings, device mesh configuration
- `tokenizer_config` block: Tokenizer loading and settings
- `task_config` block: Task type and configuration file path
- `optimizer_config` block: Learning rate, optimizer type, and schedule settings
- `training_config` block: Training parameters, checkpointing, and metrics logging
- `subset_select` block: Data subset selection and coreset algorithm parameters
- `task_config.yaml` parameters: Fine-grained control over gradient computation, selection, and curriculum learning

### Random baseline + AdamW
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:random,ratio:0.5,buffer:8} \
  optimizer_config={opt_type:adamw,learning_rate:1e-5,warmup_ratio:0.1}
```

### GREATS + ASGO
```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:greats,ratio:0.5,buffer:8} \
  optimizer_config={opt_type:asgo,learning_rate:1e-5,warmup_ratio:0.1}
```

### COLM eval using restored checkpoint
```bash
python -m tunix.cli.generate cli/base_config.yaml \
  inference_restore_step=100
```

---

If needed, add a project-local task config file next (for `task_config.config`) and keep per-experiment command templates in this README.
