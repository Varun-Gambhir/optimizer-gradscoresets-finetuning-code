# Configuration Parameter Reference

This document provides comprehensive documentation of all configuration parameters in `base_config.yaml` and `task_config.yaml`. Every parameter can be overridden at runtime from the CLI using `key=value` syntax (e.g., `optimizer_config.opt_type=asgo`) or from environment variables prefixed with `T_`.

---

## `base_config.yaml`

This is the master config file that controls every aspect of model loading, training, evaluation, and subset selection.

---

### `model_config` block

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `model_name` | string | `"llama3.1-8b"` | Identifies which model architecture to load. Format is `{family}-{size}`, e.g. `"gemma2-9b"`, `"qwen2.5-3b"`, `"llama3.1-70b"`. This string selects the right model class internally. |
| `model_source` | string | `"huggingface"` | Where to download weights from. Options: `"huggingface"`, `"kaggle"`, `"gcs"`, `""` (empty = local only). |
| `model_id` | string | path | Local filesystem path or HuggingFace model ID from which the model is loaded. For local use, this should point to the downloaded model directory. |
| `model_download_path` | string | path | Directory containing the `.safetensors` weight files. Can be the same as `model_id` for local models. |
| `rng_seed` | int | `0` | Integer seed passed to `nnx.Rngs` to control all random state — weight initialization, dropout, data shuffling during model init. |
| `model_display` | bool | `false` | If `true`, prints a human-readable summary of the model's parameter tree to stdout at startup. Useful for debugging but slow for large models. |
| `intermediate_ckpt_dir` | string | `"/tmp/intermediate_ckpt/"` | Temporary directory used when converting Gemma/Gemma2 weights downloaded from Kaggle into the internal NNX format. Not used for HuggingFace downloads. |

**`lora_config` sub-block:**

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `module_path` | string (regex) | `".*q_einsum\|.*kv_einsum\|.*gate_proj\|.*down_proj\|.*up_proj"` | A regex pattern matched against every parameter path in the model. Only parameters whose path matches this pattern receive LoRA adapters. Change this to restrict LoRA to fewer layers (e.g. `".*q_einsum"` for query-only). |
| `rank` | int | `16` | The inner dimension `r` of the LoRA decomposition. Higher values = more expressive adapters = more memory and compute. Common values: 4, 8, 16, 32, 64. |
| `alpha` | float | `2.0` | LoRA scaling factor. The effective update is scaled by `alpha / rank`. A typical heuristic is to set `alpha = rank` or `alpha = 2*rank`. Larger alpha amplifies the adapter's influence. |
| `weight_qtype` | string | `"nf4"` | Quantization format for the frozen base weights. `"nf4"` = 4-bit NormalFloat (QLoRA-style). Options depend on the model backend. Set to `""` or `"bfloat16"` to disable quantization. |
| `tile_size` | int | `256` | Block size for the NF4 quantization tiling. Larger tiles are faster but slightly less accurate. `256` is the standard NF4 block size. |

**`mesh` sub-block:**

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `shape` | string (tuple) | `"(2,2)"` | Device mesh dimensions as a string-encoded tuple. `"(2,2)"` means a 2×2 mesh of 4 devices total. For a single-node 4-GPU setup with tensor parallelism only: `"(4,1)"`. For FSDP-only: `"(1,4)"`. |
| `axis_names` | string (tuple) | `"('fsdp','tp')"` | Names for each mesh axis, matching the shape. `fsdp` = fully-sharded data parallelism axis, `tp` = tensor parallelism axis. These names are referenced by the `data_sharding_axis` in training config and by the sharding annotations in model weights. |

---

### `tokenizer_config` block

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `tokenizer_path` | string | path | Filesystem path to the tokenizer files (the folder containing `tokenizer.json`, `tokenizer_model`, or `vocab.json` depending on type). |
| `tokenizer_type` | string | `"huggingface"` | Which tokenizer library to use. Options: `"huggingface"` (loads via HuggingFace `AutoTokenizer`), `"sentencepiece"` (loads `.model` file directly). |
| `add_bos` | bool | `True` | Whether to prepend the beginning-of-sequence token to every input. Should match what the model was pretrained with. Most modern models expect BOS. |
| `add_eos` | bool | `True` | Whether to append the end-of-sequence token. Needed during training so the model learns to terminate sequences. |

---

### `task_config` block

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `task` | string | `"ift"` | High-level task type. `"ift"` = instruction fine-tuning (causal LM loss on assistant responses). `"retrieval"` = embedding/contrastive training. Controls which loss function and data pipeline are used in `peft_main.py`. |
| `config` | string | `""` | Path to the `task_config.yaml` file (see section below). Must be a valid file path. If empty, training will error at startup. |

---

### Dataset parameters (top-level)

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `dataset_name` | string | `"Helsinki-NLP/opus-100"` | HuggingFace dataset identifier or one of the special names: `"TIGER-Lab/MathInstruct"`, `"meta-math/MetaMathQA"`, `"zjunlp/Mol-Instructions"`, `"greats"`. Each triggers a different data pipeline in `ift_dataset.py`. |
| `cache_dir` | string | `""` | HuggingFace datasets cache directory. Leave empty to use the default `~/.cache/huggingface`. Useful on clusters where the home directory has limited quota. |
| `batch_size` | int | `16` | Number of examples per training step per device. The effective global batch size is `batch_size × number_of_data_parallel_devices`. |
| `eval_batch_size` | int | `16` | Batch size used during evaluation forward passes. Can be larger than `batch_size` since no gradients are stored. |
| `eval_split` | float | `0.1` | Fraction of training data to hold out for evaluation when no separate eval set is defined. Value between 0 and 1. |
| `remat_bs` | int | `16` | Rematerialization (gradient checkpointing) chunk size. Controls the tradeoff between memory and recomputation. Lower values use less memory but recompute more activations during the backward pass. |
| `num_batches` | int | `3738` | Total number of training batches per epoch. Used for scheduling purposes. Set this to `len(train_dataset) // batch_size`. |
| `max_target_length` | int | `256` | Maximum token sequence length for input and output combined. Sequences are padded or truncated to this length. Longer values consume quadratically more attention memory. |
| `num_train_epochs` | int | `1` | Number of full passes over the training dataset. Interacts with `max_steps`: training stops at whichever limit is reached first. |
| `train_fraction` | float | `1.0` | Fraction of the training dataset to actually use. Set to `0.5` to train on a random 50% of the data. Useful for ablations. |
| `num_test_batches` | int | `100` | Number of evaluation batches to use when computing validation loss during training. Limits eval time. |
| `inference_restore_step` | int | `-1` | Which checkpoint step to restore when running inference via `cli/generate.py`. `-1` means load the latest available checkpoint. Set to a specific step number to load an earlier checkpoint. |

---

### `optimizer_config` block

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `learning_rate` | float | `1e-5` | Base learning rate passed to the optimizer. The actual LR at each step is determined by the schedule. |
| `opt_type` | string | `"adamw"` | Which optimizer to use. Options: `"adamw"` (standard AdamW), `"sgd"`, `"muon"`, `"asgo"` (Adaptive Shampoo-Gradient with full Gram matrix), `"dasgo"` (diagonal Gram approximation), `"shampoo"` (full Kronecker-factored). ASGO, DASGO, and Shampoo are implemented in `cli/optax_ext.py` and only work on 2D+ parameters. |
| `schedule_type` | string | `"warmup_cosine_decay_schedule"` | Learning rate schedule. Passed to optax. Common options: `"warmup_cosine_decay_schedule"`, `"linear_schedule"`, `"constant_schedule"`. |
| `value` | float | `1e-5` | The constant LR value, used only when `schedule_type` is `"constant_schedule"`. |
| `peak_value` | float | `3e-5` | The maximum LR reached at the end of the warmup phase, for cosine/warmup schedules. |
| `init_value` | float | `0.0` | The LR at the very start of training (step 0), before warmup begins. |
| `end_value` | float | `0.0` | The LR at the end of training, after the decay phase completes. |
| `warmup_ratio` | float | `0.1` | Fraction of total training steps devoted to linear warmup. `0.1` means the first 10% of steps ramp from `init_value` to `peak_value`. |
| `b1` | float | `0.9` | Adam first moment exponential decay (momentum). Controls smoothing of the gradient. |
| `b2` | float | `0.99` | Adam second moment exponential decay. Controls smoothing of the squared gradient (effective learning rate adaptation). |
| `weight_decay` | float | `0.1` | L2 regularization coefficient for AdamW. Applied only to non-bias, non-norm parameters. |
| `warmup_steps` | int | `5` | Explicit warmup step count. If `warmup_ratio` is also set, the ratio takes precedence in the `peft_main.py` schedule builder. |
| `decay_steps` | int | `10` | Explicit decay step count used by `linear_schedule`. |
| `max_grad_norm` | float | `0.1` | Gradient clipping threshold. Gradients with global L2 norm exceeding this value are rescaled. Critical for stability, especially in early training. |

---

### `training_config` block

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `eval_every_n_steps` | int | `2` | How often (in training steps) to run a validation pass and log eval metrics. Lower values give more granular loss curves but slow training due to eval overhead. |
| `max_steps` | int | `10` | Hard cap on the total number of training steps. Training stops when this is reached even if epochs are not complete. **This is currently set very low (10) — increase to 1000+ for real experiments.** |
| `gradient_accumulation_steps` | int | `1` | Number of forward+backward passes to accumulate before performing an optimizer step. Effective batch size = `batch_size × gradient_accumulation_steps`. Set to >1 when GPU memory prevents large batch sizes. |
| `checkpoint_root_directory` | string | results path | Root directory for Orbax checkpoints. Each saved checkpoint creates a numbered subdirectory here. **Must exist before training starts.** |
| `data_sharding_axis` | list | `["fsdp"]` | Which mesh axis to shard data across. Should match one of the axis names in `mesh.axis_names`. |
| `max_inflight_computations` | int | `2` | Maximum number of asynchronous device computations that can be in-flight simultaneously. Higher values improve throughput but increase memory usage. |

**`checkpointing_options` sub-block:**

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `max_to_keep` | int | `8` | Maximum number of checkpoint versions to retain on disk. When this is exceeded, the oldest checkpoint is deleted. |
| `save_interval_steps` | int | `32` | How many training steps between each checkpoint save. **If `max_steps < save_interval_steps`, no interval-based checkpoint fires** — only the final forced save at the end of `close()`. |

**`metrics_logging_options` sub-block:**

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `log_dir` | string | results path | Directory where TensorBoard event files and W&B syncs are written. Must exist before training. |
| `flush_every_n_steps` | int | `20` | How often buffered scalar metrics are flushed to disk. **If `max_steps < flush_every_n_steps`, no flush fires mid-training.** Only flushed at `close()` if training completes. |

**`profiler_options` sub-block:**

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `log_dir` | string | results path | Where JAX/XLA profiler traces are written. These can be visualized in TensorBoard's Profile tab. |
| `skip_first_n_steps` | int | `1` | Skip profiling for this many steps at the start, to avoid capturing the one-time JIT compilation overhead which would distort the profile. |
| `profiler_steps` | int | `9` | How many consecutive steps to profile after the skip window. |

---

### `subset_select` block

This block controls the data coreset / subset selection algorithm, which is the core research contribution of this codebase.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `True` | Master switch for subset selection. If `false`, all examples in every batch are used for training (equivalent to `mode: full`). |
| `buffer` | int | `8` | How many training batches to accumulate into a large pool before selection. For example, with `batch_size=16` and `buffer=8`, a pool of 128 examples is collected, subset selection picks `ratio × 128` of them, then those are split back into mini-batches for gradient updates. Larger buffers give the selection algorithm more candidates to choose from. |
| `ratio` | float | `0.5` | Fraction of the buffer to select. `0.5` means 50% of buffered examples are kept. The selected examples are used for the actual parameter update. |
| `mode` | string | `"random"` | Which selection algorithm to use. Options: `"full"` (use all data, no selection), `"random"` (random permutation), `"gradnorm"` (highest gradient norm examples), `"facloc"` (facility location submodular maximization), `"greats"` (gradient-alignment-based selection from the GREATS paper), `"joint"` (domain-aware GREATS with curriculum learning and APGD solver). |

---

## `task_config.yaml`

This file is the fine-grained control panel for subset selection, gradient computation, and curriculum learning. It is loaded by `peft_main.py` and passed as `task_config.config` into the entire training pipeline, where it is accessed everywhere as `task_cfg`.

---

### Top-level fields

| Parameter | Type | Example | What it does |
|---|---|---|---|
| `task` | string | `"ift"` | Should match the `task` in `base_config.yaml`. Redundant but used as a sanity check. |
| `mode` | string | `"contrastive"` | Used only for retrieval tasks. Ignored for IFT. Options: `"contrastive"`. |
| `lamb` | list | `[1.0, 1e-2, 512]` | For retrieval tasks, controls loss weighting: `[start_value, end_value, breakoff_step]`. The loss coefficient decays from `start` to `end` over `breakoff` steps. Ignored for IFT. |

---

### `config.gradsapprox` block

Controls the gradient approximation network (`GradApprox`) that optionally learns to estimate gradient similarities cheaply.

| Parameter | Type | Example | What it does |
|---|---|---|---|
| `lr_head` | float | `5e-5` | Learning rate for the `GradApprox` head network's optimizer (`optimizer_head`). This is a separate small network trained alongside the main model to approximate per-example gradient fingerprints. |
| `gradapprox_warmup` | int | e.g. `100` | Number of steps before the gradient approximation network begins contributing. During warmup, exact gradients are used. After warmup, the approximation can kick in. |
| `grads_approx` | string | `"none"` | Which approximation mode to use. `"none"` means always compute exact per-example gradients via `nnx.vmap` over `value_and_grad`. Any other value would trigger an approximation path (currently raises `ValueError` — only `"none"` is implemented). |

---

### `config.grads` block

Controls how per-example gradients are computed and processed before being used for selection.

| Parameter | Type | Example | What it does |
|---|---|---|---|
| `grad_layer` | string or int | e.g. `"last"` or `8` | Which LoRA layer's gradients to use as the per-example fingerprint. Computing gradients for all layers is expensive, so this restricts to a specific layer. `"last"` uses the final LoRA layer. A number selects a specific layer index. |
| `normalize` | bool | `False` | If `true`, per-example gradient vectors are L2-normalized before computing similarities or scores. This makes selection direction-aware rather than magnitude-aware, often improving coreset quality. |
| `lowpass_adam` | bool | `False` | If `true`, applies Adam-style exponential moving average smoothing to the raw gradients before selection (the `update_moments` function). This is called "lowpass" because it dampens high-frequency gradient noise. Requires maintaining first and second moment buffers. |
| `dimred` | bool | `False` | If `true`, applies dimensionality reduction to the gradient vectors via FFT sketching (`dimred_fft`). Reduces the cost of computing the Gram matrix when gradients are very high-dimensional. Cannot be used simultaneously with `lowpass_adam`. |
| `val_smooth` | string | `"val"` | Controls which momentum state is used for the validation anchor gradients. `"val"` uses the validation batch's own computed moments. `"train"` reuses the training batch's moments for the validation pass, saving one set of moment buffers. |

---

### `config.subsel` block

Fine-grained control over how the selection algorithm behaves at each step.

| Parameter | Type | Example | What it does |
|---|---|---|---|
| `val_anchors` | bool | `True` | Whether to use a separate validation batch as the "anchor" for GREATS/joint selection. When `true`, the algorithm selects training examples that are most aligned with the validation gradient direction. When `false`, training examples are used as their own anchors (self-referential selection). |
| `anchorbs` | int | e.g. `32` | Batch size for the validation anchor batch fetched at each step from `dev_ds`. Only relevant when `val_anchors: true`. Larger values give a more stable validation signal but cost more compute. |
| `domainwise` | bool | `False` | If `true`, selection is performed independently within each source domain (e.g. each math topic). Each domain gets its own budget allocation via `select_per_class`. When `false`, selection is done globally across all domains. |
| `eval_source` | string | `"train"` or `"colm"` | Where the evaluation dataset comes from. `"train"` splits a random fraction of the training data for eval. `"colm"` uses the COLM few-shot benchmark examples as the eval set. This controls what is passed as `eval_ds` during training. |
| `dev_source` | string | `"train"`, `"eval"`, or `"colm"` | Where the development/anchor dataset comes from. `"train"` creates a separate split. `"eval"` reuses the same eval set. `"colm"` uses COLM benchmark examples. This is the dataset used for GREATS/joint validation anchors. |
| `minority_full` | bool | `False` | If `true`, examples from minority classes (defined by `minority_classes`) are always included in full in the selected subset, regardless of the selection budget. This prevents rare categories from being entirely dropped by the selection algorithm. |
| `minority_classes` | list of ints | e.g. `[3, 7]` | Class IDs (source domain indices) that should always receive full inclusion when `minority_full: true`. These are the integer IDs assigned by `src2id` during dataset loading. |
| `apdg_iters` | int | e.g. `10` | Number of Accelerated Projected Gradient Descent iterations inside the `joint_subsel` solver. Used only when `mode: "joint"`. More iterations give a better optimization solution but cost more compute per selection step. |

---

### `config.curriculum` block

Controls the curriculum learning component used in joint mode, which dynamically adjusts per-domain selection budgets based on how much each domain is currently benefiting from training.

| Parameter | Type | Example | What it does |
|---|---|---|---|
| `beta` | float | e.g. `0.9` | Exponential moving average coefficient for updating the per-class gain signal. `gain = beta × delta_utility + (1 - beta) × gain`. Higher beta makes the curriculum more conservative/slow-changing. |
| `lamb` | float | e.g. `0.1` | Strength of the curriculum's influence on per-class budget adjustment. Higher values allow larger deviations from the proportional budget allocation. `lamb=0` effectively disables curriculum and falls back to proportional allocation. |

---

### `config.domain_weights` block

Allows externally-specified per-domain sampling weights to override uniform or proportional sampling during data loading.

| Parameter | Type | Example | What it does |
|---|---|---|---|
| `enabled` | bool | `False` | Master switch for weighted domain sampling. When `false`, the DataLoader uses uniform shuffling. When `true`, a `WeightedRandomSampler` is constructed using the weights from `weights_dir`. |
| `weights_dir` | string | path to a `.json` file | Path to a JSON file containing a `"train_domain_weights"` key whose value is a dictionary mapping domain name to sampling weight. For example: `{"train_domain_weights": {"algebra": 0.6, "calculus": 0.4}}`. The domain names must match the keys in `src2id`. |

---

### `config.wordydev` block

| Parameter | Type | Example | What it does |
|---|---|---|---|
| `wordydev` | string | `""`, `"dev"`, `"eval"`, or `"deveval"` | Controls whether answer choice text is appended to MMLU prompts when building the dev and eval datasets. `"dev"` appends choices to dev prompts only, `"eval"` to eval prompts only, `"deveval"` to both. Empty string means no appending. This changes what the model sees during gradient-based anchor computation. |

---

## Quick Reference: Common Configuration Patterns

### LoRA Fine-tuning with GREATS subset selection
```yaml
lora_enabled: true
optimizer_config:
  opt_type: adamw
  learning_rate: 1e-5
  warmup_ratio: 0.1
subset_select:
  enabled: true
  mode: greats
  ratio: 0.5
  buffer: 8
```

### Full fine-tuning with AdamW optimizer
```yaml
lora_enabled: false
optimizer_config:
  opt_type: adamw
  learning_rate: 5e-5
  warmup_ratio: 0.1
subset_select:
  enabled: false
  mode: full
```

### Random baseline
```yaml
subset_select:
  enabled: true
  mode: random
  ratio: 0.5
  buffer: 8
```

### Joint mode with curriculum learning
```yaml
subset_select:
  enabled: true
  mode: joint
  ratio: 0.5
  buffer: 8
```
