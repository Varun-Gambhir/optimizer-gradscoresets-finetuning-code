# Optimizer Selection for Gradient Coresets

> Training and evaluation utilities for SFT/PEFT, subset selection (including GREATS), and COLM-style math evaluation.

---

## Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Project Structure](#2-project-structure)
- [3. Repo Notes Before Running](#3-repo-notes-before-running)
- [4. LoRA Configuration](#4-lora-configuration)
- [5. Running the Main Training Script](#5-running-the-main-training-script)
- [6. Configuring Optimizer and Subset Selection](#6-configuring-optimizer-and-subset-selection)
- [7. Configuration Quick Guide](#7-configuration-quick-guide)
- [8. Run Training](#8-run-training)
- [9. Choose Optimizer](#9-choose-optimizer)
- [10. COLM Evaluation](#10-colm-evaluation)
- [11. Common Issues and Fixes](#11-common-issues-and-fixes)
- [12. Minimal Repro Commands](#12-minimal-repro-commands)
- [13. Configuration Parameter Reference](#13-configuration-parameter-reference)

---

## 1. Environment Setup

### System Requirements

| Requirement | Version |
|---|---|
| OS | Linux |
| Python | 3.10+ |
| GPU | CUDA-compatible with JAX |

### Create and Activate Conda Environment

```bash
conda create -n myenv python=3.10.12
conda activate myenv
```

### Install Dependencies

From the repository root, run:

```bash
python install_packages.py
```

This installs all required dependencies — `jax`, `flax`, `optax`, `transformers`, and more.

> **HuggingFace Token** — If you plan to download models or tokenizers from HuggingFace, set your token:
> ```bash
> export HF_TOKEN=<your_hf_token>
> ```

---

## 2. Project Structure

```
tunix/
│
├── cli/                               ← Training entry points and config
│   ├── base_config.yaml               ← Main experiment config (all hyperparameters)
│   ├── task_config_baseline.yaml      ← Task-specific settings (IFT, retrieval)
│   ├── peft_main.py                   ← Main training launcher
│   ├── generate.py                    ← Inference / COLM evaluation launcher
│   ├── grpo_main.py                   ← GRPO RL training launcher (Not Used Anymore)
│   ├── config.py                      ← HyperParameters config loader class
│   ├── loss.py                        ← Loss functions (IFT, retrieval, contrastive)
│   ├── optax_ext.py                   ← Custom optimizers (ASGO, DASGO, Shampoo)
│   ├── eval.py                        ← IR / MTEB evaluation utilities
│   ├── beir.py                        ← BEIR benchmark helpers
│   └── utils/                         ← Config and training utilities
│
├── generate/                          ← Inference engine
│   ├── sampler.py                     ← Main auto-regressive sampler
│   ├── beam_search.py                 ← Beam search implementation
│   ├── base_sampler.py                ← Abstract sampler base class
│   ├── vllm_sampler.py                ← vLLM-backed sampler
│   ├── tokenizer_adapter.py           ← Unified tokenizer wrapper
│   ├── utils.py                       ← Attention masks, padding helpers
│   └── mappings.py                    ← Weight name mapping utilities
│
├── models/                            ← Model architecture definitions
│   ├── gemma/                         ← Gemma/Gemma2 model
│   ├── gemma3/                        ← Gemma3 model
│   ├── llama3/                        ← Llama3 model
│   ├── qwen2/                         ← Qwen2.5 model
│   ├── qwen3/                         ← Qwen3 model (with MoE)
│   └── safetensors_loader.py          ← Shared concurrent safetensors loader
│
├── sft/                               ← Training infrastructure
│   ├── subset_trainer.py              ← Custom trainer with subset selection loop
│   ├── subsel_utils.py                ← All selection algorithms (GREATS, FacLoc etc.)
│   ├── peft_trainer.py                ← Base PEFT trainer
│   ├── checkpoint_manager.py          ← Orbax checkpoint save/restore
│   ├── metrics_logger.py              ← TensorBoard + W&B logging
│   ├── utils.py                       ← Attention masks, HBM monitoring
│   ├── hooks.py                       ← Training/data hook interfaces
│   ├── sharding_utils.py              ← JAX sharding helpers
│   ├── progress_bar.py                ← tqdm progress bar
│   ├── profiler.py                    ← JAX profiler wrapper
│   ├── inflight_throttler.py          ← Async compute throttling
│   ├── system_metrics_calculator.py   ← TFLOP measurement
│   └── eval/                          ← Evaluation utilities
│       ├── colm/                      ← COLM benchmark runner
│       ├── data_selection/            ← Dataset loaders
│       ├── mol/                       ← Molecule translation metrics
│       ├── mmlu_eval.py               ← Standalone MMLU evaluator
│       ├── tydiqa_eval.py             ← Standalone TydiQA evaluator
│       └── chat_templates.py          ← Prompt formatting templates
│
├── examples/
│   └── data/                          ← Data pipelines
│       ├── ift_dataset.py             ← IFT data pipeline
│       ├── retrieval_dataset.py       ← Retrieval/embedding data pipeline
│       ├── nomic_jax_dataset.py       ← Nomic embedding dataset
│       └── pretrain_dataset.py        ← Pre-training data pipeline
│
├── utils/                             ← Utility modules
│   ├── compat.py                      ← JAX/Flax version compatibility shims
│   └── container.py                   ← ModuleList compatibility shim
│
├── oss/                               ← External resource utilities
│   └── utils.py                       ← GCS, Kaggle, HuggingFace download helpers
│
└── __init__.py                        ← Public Tunix API exports
```

---

## 3. Repo Notes Before Running

| Entrypoint | File |
|---|---|
| SFT/PEFT training | `cli/peft_main.py` |
| Generation / COLM evaluation | `cli/generate.py` |
| Base configuration | `cli/base_config.yaml` |

> **Important:** `task_config.config` in `cli/base_config.yaml` must point to a valid YAML file.
> See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for detailed parameter documentation.
> `cli/peft_main.py` loads this path with OmegaConf at runtime.

---

## 4. LoRA Configuration

The training script supports both LoRA (Low-Rank Adaptation) fine-tuning and full model fine-tuning.

### Enable LoRA

```yaml
lora_enabled: true
```

### Disable LoRA (Full Fine-tuning)

```yaml
lora_enabled: false
```

---

## 5. Running the Main Training Script

From the repository root, run:

```bash
python -m tunix.cli.peft_main base_config.yaml > training.log 2>&1
```

This will:

1. Load configuration from `cli/base_config.yaml`
2. Set up the model and tokenizer
3. Initialize the selected optimizer and subset selection mode
4. Start training and save logs to `training.log`

---

## 6. Configuring Optimizer and Subset Selection

To change the optimizer and subset selection mode without modifying command-line arguments, edit `cli/base_config.yaml` directly.

### Available Optimizers

Edit `optimizer_config.opt_type` in `cli/base_config.yaml`:

| Optimizer | Description |
|---|---|
| `adamw` | Standard AdamW (recommended default) |
| `sgd` | Stochastic Gradient Descent |
| `muon` | Muon optimizer |
| `asgo` | Adaptive Shampoo-Gradient (full Gram matrix) |
| `dasgo` | Diagonal Gram approximation |
| `shampoo` | Full Kronecker-factored preconditioner |

### Available Subset Selection Modes

Edit `subset_select.mode` and other subset selection parameters in `cli/base_config.yaml`:

| Mode | Description |
|---|---|
| `full` | Use all data, no selection |
| `random` | Random permutation baseline |
| `gradnorm` | Select highest gradient norm examples |
| `facloc` | Facility location submodular maximization |
| `greats` | Gradient-alignment-based selection (GREATS paper) |
| `joint` | Domain-aware GREATS with curriculum learning and APGD solver |

Additional parameters:
- `ratio` — fraction of data to select (0.0 to 1.0)
- `buffer` — number of batches to pool before selection

### Example `base_config.yaml` Setup

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

---

## 7. Configuration Quick Guide

Key fields in `cli/base_config.yaml` that you will commonly change:

| Parameter | Options / Notes |
|---|---|
| `optimizer_config.opt_type` | `adamw`, `sgd`, `muon`, `asgo`, `dasgo`, `shampoo` |
| `subset_select.enabled` | `true` / `false` |
| `subset_select.mode` | `full`, `random`, `gradnorm`, `facloc`, `greats`, `joint` |
| `subset_select.ratio` | Float between 0.0 and 1.0 |
| `subset_select.buffer` | Number of batches to buffer before selection |
| `task_config.config` | Path to your `task_config.yaml` file |
| `training_config.max_steps` | Total training steps |
| `batch_size` | Examples per step |
| `lora_enabled` | `true` for LoRA, `false` for full fine-tuning |

---

## 8. Run Training

All commands below are run from the repository root.

### Base Command

```bash
python -m tunix.cli.peft_main cli/base_config.yaml
```

### A) Baseline Runs

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

### B) GREATS Run

```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:greats,ratio:0.5,buffer:8}
```

### C) Joint Mode

```bash
python -m tunix.cli.peft_main cli/base_config.yaml \
  subset_select={enabled:true,mode:joint,ratio:0.5,buffer:8}
```

---

## 9. Choose Optimizer

Set via `optimizer_config.opt_type` in the config file, or override from the CLI.

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

---

## 10. COLM Evaluation

The default behavior of `cli/generate.py` is to run `COLM_eval(...)`.

```bash
python -m tunix.cli.generate cli/base_config.yaml
```

**What this does:**

1. Loads model and tokenizer
2. Restores LoRA checkpoint from `training_config.checkpoint_root_directory` and `inference_restore_step`
3. Runs evaluation on COLM datasets: `numglue`, `mmlu_mathematics`, `gsm8k`, `svamp`, `simuleq`, `deepmind`, `aqua`, `sat`
4. Writes per-dataset JSONL outputs and a summary CSV under the results directory

> **Paths used in the current COLM flow:**
> - Dataset root: `/home/temp/CoLM/math_eval/dataset`
> - Output root: `/home/temp/tunix/examples/sft/mtnt/results/<exp_name>/<ckpt_num>`
>
> Ensure these paths exist or update them in `cli/generate.py` before running.

---

## 11. Common Issues and Fixes

### Missing HuggingFace Token

**Symptom:** Tokenizer or model load fails.

```bash
export HF_TOKEN=<your_hf_token>
```

---

### `task_config.config` is Empty

**Symptom:** OmegaConf load error at training startup.

**Fix:** Set `task_config.config` in `cli/base_config.yaml` to a real YAML file path.

---

### Unknown Optimizer Type

**Symptom:** `Unknown optimizer type: ...`

**Fix:** Use one of the supported types: `adamw`, `sgd`, `muon`, `asgo`, `dasgo`, `shampoo`.

---

### COLM Path Errors

**Symptom:** Dataset not found or output path issues.

**Fix:** Create or adjust the following paths:
- `/home/temp/CoLM/math_eval/dataset`
- `/home/temp/tunix/examples/sft/mtnt/results/...`

---

## 12. Minimal Repro Commands

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

### COLM Eval Using Restored Checkpoint
```bash
python -m tunix.cli.generate cli/base_config.yaml \
  inference_restore_step=100
```

---

## 13. Configuration Parameter Reference

For detailed documentation of all configuration parameters across `base_config.yaml` and `task_config.yaml`, see [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

This reference covers:

| Section | Description |
|---|---|
| `model_config` | Model loading, LoRA settings, device mesh configuration |
| `tokenizer_config` | Tokenizer loading and settings |
| `task_config` | Task type and configuration file path |
| `optimizer_config` | Learning rate, optimizer type, and schedule settings |
| `training_config` | Training parameters, checkpointing, and metrics logging |
| `subset_select` | Data subset selection and coreset algorithm parameters |
| `task_config.yaml` parameters | Fine-grained control over gradient computation, selection, and curriculum learning |

---

> If needed, add a project-local task config file (for `task_config.config`) and keep per-experiment command templates in this README.