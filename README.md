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

## 2) Repo Notes Before Running

- Main SFT/PEFT training entrypoint: `cli/peft_main.py`
- Main generation/eval entrypoint (currently calls COLM eval by default): `cli/generate.py`
- GRPO entrypoint: `cli/grpo_main.py`
- Base config: `cli/base_config.yaml`

Important:
- `task_config.config` in `cli/base_config.yaml` must point to a valid YAML file.
- `cli/peft_main.py` loads this path with OmegaConf at runtime.

## 3) LoRA Configuration

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

## 4) Running the Main Training Script

From the repository root, run:
```bash
python -m tunix.cli.peft_main base_config.yaml > training.log 2>&1
```

This will:
1. Load configuration from `cli/base_config.yaml`
2. Set up the model and tokenizer
3. Initialize the selected optimizer and subset selection mode
4. Start training and save logs to `training.log`

## 5) Configuring Optimizer and Subset Selection

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

## 6) Configuration Quick Guide

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

## 7) Run Training

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

## 8) Choose Optimizer

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

## 9) COLM Evaluation

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

## 10) GRPO Training (Optional)

```bash
python -m tunix.cli.grpo_main cli/base_config.yaml
```

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
