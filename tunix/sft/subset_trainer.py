from collections.abc import Iterable
import contextlib
import dataclasses
import time
from typing import Any, Callable, Concatenate, Dict, List, ParamSpec, Tuple

from absl import logging
import flax
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import jax.sharding as shd
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import jaxtyping
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.sft import checkpoint_manager
from tunix.sft import hooks
from tunix.sft import inflight_throttler
from tunix.sft import metrics_logger
from tunix.sft import profiler
from tunix.sft import progress_bar
from tunix.sft import sharding_utils
from tunix.sft import system_metrics_calculator
from tunix.sft import utils
from flax.core import FrozenDict
from functools import partial
from tunix.sft import subsel_utils

_ModelInputT = Dict[str, ArrayLike]
P = ParamSpec("P")
# class SubsetGradParam(nnx.Param): pass
from flax.nnx import rnglib, variablelib
import typing as tp
A = tp.TypeVar('B')

class SubsetGradParam(variablelib.Param[A]):
  pass
# @dataclasses.dataclass(slots=True, kw_only=True)
@flax.struct.dataclass(frozen=True)
class TrainingConfig:
  """Configuration for the trainer."""

  eval_every_n_steps: int
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None

  # If set, the checkpoints will be saved to this path. Checkpoints
  # contains the model params and the train data iterator state.
  checkpoint_root_directory: str | None = None
  # Checkpoint configurations. If None, the default options will be used.
  checkpointing_options: ocp.CheckpointManagerOptions | None = None

  # Configs for the metrics logger.
  metrics_logging_options: metrics_logger.MetricsLoggerOptions | None = None

  # Configs for the profiler.
  profiler_options: profiler.ProfilerOptions | None = None

  data_sharding_axis: Tuple[str, ...] = ("fsdp",)

  # Controls how many train_steps can be scheduled ahead of time.
  max_inflight_computations: int = 2

  # Prefix for metric names for logging. Not sticking it in
  # `metrics_logging_options` because the latter is optional.
  metric_prefix: str = ""

  # Progress bar description.
  pbar_description: str | None = "Training"

  def get_with_default(self, key: str, default: Any) -> Any:
    val = getattr(self, key)
    if val is None:
      return default
    return val


@flax.struct.dataclass(frozen=True)
class TrainingInput:
  # Input tokens provided to the model.
  input_tokens: jax.Array | np.ndarray

  # A mask that determines which input tokens are valid.
  input_mask: jax.Array | np.ndarray


@dataclasses.dataclass(slots=True, kw_only=True)
class MetricsBuffer:
  """Metrics collected for a specific step.

  Attributes:
    step: The training step number.
    losses: A list of loss values recorded within this step (e.g., across
      gradient accumulation steps).
    step_time_deltas: A list of time deltas for each computation within this
      step.
    additional_metrics: Dictionary for storing additional metrics. The key is
      the metric name, and the value is a tuple containing a list of metric
      values and a callable to aggregate them.
  """

  step: int
  losses: List[ArrayLike]
  step_time_deltas: List[float]
  additional_metrics: Dict[str, ArrayLike] = dataclasses.field(default_factory=dict)

  @property
  def loss(self):
    """Returns the mean of the recorded losses for the step."""
    return np.mean(self.losses)

  @property
  def step_time_delta(self):
    """Returns the mean of the recorded step time deltas for the step."""
    return np.mean(self.step_time_deltas)


def _calculate_global_batch_size(train_example: Any) -> int:
  """Calculates the global batch size from a training example.

  Args:
    train_example: A training example, which can be a dataclass, a dict, or an
      object with attributes.

  Returns:
    The global batch size.

  Raises:
    TypeError: If the batch size cannot be determined from the training example.
  """
  if dataclasses.is_dataclass(train_example):
    attributes = dataclasses.asdict(train_example)
  elif isinstance(train_example, dict):
    attributes = train_example
  else:
    attributes = vars(train_example)

  for field_value in attributes.values():
    if isinstance(field_value, (jax.Array, np.ndarray)):
      # Assume the first array we find has the batch dimension.
      return field_value.shape[0]

  raise TypeError(
      "Could not automatically determine batch size. No JAX or NumPy "
      "array found in the training example."
  )


class MLP2(nnx.Module):
  """MLP2 module."""

  def __init__(
      self,
      embed_dim,
      hidden_dim,
      out_dim,
      param_dtype,
      dtype,
      *,
      rngs: nnx.Rngs,
  ):
    # nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape)
    # kernel_init_fn = nnx.initializers.zeros_init()
    # self.gate_proj = nnx.Linear(
    #     in_features=embed_dim,
    #     out_features=hidden_dim,
    #     use_bias=False,
    #     rngs=rngs,
    #     param_dtype=param_dtype,
    #     dtype=dtype,
    #     # kernel_init=kernel_init_fn,
    # )
    self.up_proj = nnx.Linear(
        in_features=embed_dim,
        out_features=hidden_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
        dtype=dtype,
        # kernel_init=kernel_init_fn,
    )
    self.down_proj = nnx.Linear(
        in_features=hidden_dim,
        out_features=out_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
        dtype=dtype,
        # kernel_init=kernel_init_fn,
    )

  @jax.named_scope('feed_forward123')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    activations = nnx.silu(self.up_proj(x))
    outputs = self.down_proj(activations)
    return outputs

class Linear(nnx.Module):
  """MLP2 module."""

  def __init__(
      self,
      embed_dim,
      out_dim,
      param_dtype,
      dtype,
      *,
      rngs: nnx.Rngs,
  ):
    self.proj = nnx.Linear(
        in_features=embed_dim,
        out_features=out_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
        dtype=dtype,
        # kernel_init=kernel_init_fn,
    )

  @jax.named_scope('feed_forward123')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    outputs = self.proj(x)
    return outputs.astype(jnp.float32)

class GradApprox2(nnx.Module):
  def __init__(
      self,
      config,
      *,
      rngs: nnx.Rngs,
  ):
    # nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape)
    # kernel_init_fn = nnx.initializers.zeros_init()
    self.mlp1 = MLP2(
        rngs=rngs,
        embed_dim=config.embed_dim,
        hidden_dim=config.embed_dim,
        out_dim=config.embed_dim,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.mlp2 = MLP2(
        rngs=rngs,
        embed_dim=config.embed_dim,
        hidden_dim=config.embed_dim,
        out_dim=config.embed_dim,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.ln = nnx.LayerNorm(config.embed_dim, rngs=rngs)
    
  @jax.named_scope('grad_approx')
  # def __call__(self, x: jaxtyping.ArrayLike, moment1: jaxtyping.ArrayLike, alpha=1) -> jaxtyping.Array:
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    out1 = self.mlp1(x)
    out1 = self.ln(out1)
    out1 = out1 + x
    out = self.mlp2(out1)
    # out = alpha*out + (1-alpha)*moment1
    return out

import jax
import jax.numpy as jnp
from flax import nnx

class LRL(nnx.Module):
  """Linear-(SiLU)-Linear low-rank: D -> r -> D"""
  def __init__(self, dim: int, rank: int, out_dim: int, *, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(dim, rank, rngs=rngs)
    self.fc2 = nnx.Linear(rank, out_dim, rngs=rngs)

  def __call__(self, x):
    return self.fc2(jax.nn.silu(self.fc1(x)))

class LRL_joint(nnx.Module):
  """Linear-(SiLU)-Linear low-rank: D -> r -> D"""
  def __init__(self, dim: int, rank: int, out_dim: int, k:int, *, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(dim, rank, rngs=rngs)
    self.fc2 = nnx.Linear(rank, k*out_dim, rngs=rngs)
    self.dim = dim
    self.out_dim = out_dim
    self.k = k

  def __call__(self, x):
    bs = x.shape[0]
    return self.fc2(jax.nn.silu(self.fc1(x))).reshape((bs, self.k, self.out_dim))


class GradApproxBig(nnx.Module):
  def __init__(
      self,
      config,
      *,
      rngs: nnx.Rngs,
      use_heads: bool = False,  # compile-time constant
      num_heads: int = 64,       # compile-time constant
  ):
    hidden_dim = 1024
    mlp_out_dim = config.embed_dim
    out_dim = config.embed_dim
    # D must be known at init time (and must match x.shape[-1] at runtime)
    # hidden_dim = 1024
    # mlp_out_dim = 1024
    out_dim = config.embed_dim
    D = mlp_out_dim
    # low-rank chosen purely from D and k (no extra flags)
    # r = max(1, D // (4 * max(1, self.k)))
    r = 256

    self.mlp1 = MLP2(
        rngs=rngs,
        embed_dim=config.embed_dim,
        hidden_dim=hidden_dim,
        out_dim=mlp_out_dim,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    # self.mlp2 = MLP2(
    #     rngs=rngs,
    #     embed_dim=config.embed_dim,
    #     hidden_dim=hidden_dim,
    #     out_dim=out_dim,
    #     param_dtype=config.param_dtype,
    #     dtype=config.dtype,
    # )
    self.joint_proj = MLP2(
        rngs=rngs,
        embed_dim=2*mlp_out_dim,
        hidden_dim=256,
        out_dim=mlp_out_dim,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.joint = True
    self.ln = nnx.LayerNorm(mlp_out_dim, rngs=rngs)

    self.use_heads = nnx.static(use_heads)
    self.k = nnx.static(num_heads)

    if self.use_heads:
      # k separate heads, each independent params
      
      # self.heads = nnx.List([Linear(
      #   rngs=rngs, embed_dim=config.embed_dim, out_dim=config.embed_dim, param_dtype=config.param_dtype,
      #   dtype=config.dtype,) for _ in range(self.k)])

      # self.heads = nnx.List([LRL(D, r, out_dim, rngs=rngs) for _ in range(self.k)])
      
      self.heads = LRL_joint(D, r, out_dim, self.k, rngs=rngs)
    else:
      self.heads = None

  @jax.named_scope("grad_approx")
  def __call__(self, delta, x):
    # (optional) runtime safety check; remove if you hate asserts
    # assert x.shape[-1] == self.ln.features  # or == config.embed_dim

    h = self.mlp1(x)
    if self.joint:
      h = self.joint_proj(jnp.concat([delta, h], axis=1))
    # h = self.ln(h)
    # h = h + x
    # h = self.mlp2(h)  # (..., D)

    # if not self.use_heads:
    return h

    # # outs = [head(h) for head in self.heads]     # each (..., D)
    # outs = self.heads(h)     # each (..., D)
    # # outs = jnp.stack(outs, axis=-2) # (..., k, D)
    # return outs


# simple approx
import chex
class GradApprox(nnx.Module, pytree=False):
  def __init__(
      self,
      config,
      *,
      rngs: nnx.Rngs,
  ):
    hidden_dim = 1024
    # hidden_dim = config.embed_dim
    out_dim = config.embed_dim
  
    # kernel_init_fn = nnx.initializers.zeros_init()
    kernel_init_fn = nnx.initializers.xavier_normal()
    self.linear = nnx.Linear(
      rngs=rngs,
      in_features=config.embed_dim,
      out_features=hidden_dim,
      param_dtype=config.param_dtype,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
          kernel_init_fn, config.shd_config.ffw_weight_df
      ),
    )

    self.linear2 = nnx.Linear(
      rngs=rngs,
      # in_features=hidden_dim+config.embed_dim,
      in_features=hidden_dim,
      out_features=out_dim,
      param_dtype=config.param_dtype,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
          kernel_init_fn, config.shd_config.ffw_weight_df
      ),
    )
    rank = 256
    self.linear3 = nnx.Linear(
      rngs=rngs,
      # in_features=hidden_dim+config.embed_dim,
      in_features=config.embed_dim,
      out_features=rank,
      param_dtype=config.param_dtype,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
          kernel_init_fn, config.shd_config.ffw_weight_df
      ),
    )
    self.linear4 = nnx.Linear(
      rngs=rngs,
      # in_features=hidden_dim+config.embed_dim,
      in_features=rank,
      out_features=config.embed_dim,
      param_dtype=config.param_dtype,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
          kernel_init_fn, config.shd_config.ffw_weight_fd
      ),
    )

    self.final_linear = nnx.Linear(
      rngs=rngs,
      in_features=out_dim,
      out_features=config.embed_dim,
      param_dtype=config.param_dtype,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
          kernel_init_fn, config.shd_config.ffw_weight_fd
      ),
    )

    self.w = nnx.Param(
        nnx.initializers.normal()(rngs.params(), (config.embed_dim))
    )
    
   
  # def __call__(self, delta, x):
  #   Bs, D = delta.shape[0], delta.shape[1]
  #   outer = jnp.einsum("Px, Py -> Pxy", nnx.tanh(x), nnx.tanh(x)) #[B, d, d]
  #   chex.assert_shape(outer, (Bs, D, D))
  #   # outer = jnp.einsum("Px, Py -> Pxy", x,x) #[B, d, d]
  #   # outer = nnx.tanh(outer)
  #   outer = jnp.reshape(outer, (-1, D))

  #   h = self.linear(outer) # [Bd, d2]

  #   h = self.linear2(nnx.silu(h)) # [Bd, d]

  #   h = h.reshape((Bs, D, D))
  #   grads = subsel_utils.delta_jacobian(delta, h)
    
  #   return grads

  # # linearOuterJacTanh
  # WORKS
  def __call__(self, delta, x):
    Bs, D = delta.shape[0], delta.shape[1]
    outer = jnp.einsum("Px, Py -> Pxy", nnx.tanh(x), nnx.tanh(x)) #[B, d, d]
    chex.assert_shape(outer, (Bs, D, D))
    outer = jnp.reshape(outer, (-1, D))

    h = self.linear3(outer) # [Bd, D]
    h = self.linear4(h)
    h = nnx.tanh(h)
    h = h.reshape((Bs, D, D))
    grads = subsel_utils.delta_jacobian(delta, h)
    
    return grads

  # def __call__(self, delta, x):
  #   Bs, D = delta.shape[0], delta.shape[1]

  #   h = self.linear(x) # [B, d]
  #   h = nnx.tanh(h)
  #   # h = nnx.silu(h)

  #   h = self.linear2(jnp.concat([delta, h], axis=-1)) # [B, d_out]
  #   h = nnx.tanh(h)
  #   # h = nnx.silu(h)
  #   # grads = self.final_linear(jnp.concat([delta, h], axis=1)) # [B, d]
  #   # grads = self.final_linear(h) # [B, d]
  #   # grads = subsel_utils.delta_jacobian(delta, h)
    
  #   return h


  # # WORKS
  # def __call__(self, delta, x):
  #   Bs, D = delta.shape[0], delta.shape[1]

  #   h = x*self.w.value # [B, d]
  #   h = nnx.tanh(h)
  #   h = delta*h
  #   return h

  # def __call__(self, delta, x):
  #   Bs, D = delta.shape[0], delta.shape[1]

  #   h = x*self.w.value # [B, d]
  #   h = nnx.tanh(h)
  #   h = delta*h
  #   return h



class PeftTrainer:
  """PEFT trainer for LoRA. Only LoRA parameters are updated.

  Attributes:
    model: The model to train.
    config: The training config.
    optimizer: The optimizer to use. To monitor the learning rate at each step,
      use `optax.schedules.inject_hyperparams` to inject learning rate as a
      hyperparameter. For example: ``optimizer =
      optax.schedules.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule)``
    loss_fn: The loss function to use.
    eval_loss_fn: The loss function to use for evaluation.
    gen_model_input_fn: The function to generate model input from training
      input.
    checkpoint_manager: The checkpoint manager to use.
    metrics_logger: The metrics logger to use.
    is_managed_externally: Whether the trainer is managed externally.
    training_hooks: The training hooks to use.
    data_hooks: The data hooks to use.
  """

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
      optimizer_head: optax.GradientTransformation,
      optimizer_head2: optax.GradientTransformation,
      optimizer_last: optax.GradientTransformation,
      train_loss: Callable,
      eval_loss: Callable ,
      eval_fn: Callable,
      has_aux: bool,
      schedule,
      config
  ):
    self._validate_config(training_config)
    self.model = model
    self.config = training_config
    self.fullConfig = config
    self.schedule = schedule
    self._lora_enabled = utils.is_lora_enabled(self.model)
    if training_config.gradient_accumulation_steps is not None:
      optimizer = optax.MultiSteps(
          optimizer, training_config.gradient_accumulation_steps
      )
      optimizer_last = optax.MultiSteps(
          optimizer_last, training_config.gradient_accumulation_steps
      )
    if self._lora_enabled:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.LoRAParam)
      self.optimizer_last = nnx.Optimizer(self.model, optimizer_last, wrt=partial(subsel_utils.grad_filter, [27]))
    else:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.Param)
    self.project = GradApprox(model.config, rngs=nnx.Rngs(params=0))
    self.project2 = GradApprox(model.config, rngs=nnx.Rngs(params=0))
    self.optimizer_head = nnx.Optimizer(self.project, optimizer_head, wrt=nnx.Param)
    self.optimizer_head2 = nnx.Optimizer(self.project2, optimizer_head2, wrt=nnx.Param)
    self.loss_fn = train_loss
    self.eval_loss_fn = eval_loss
    self.gen_model_input_fn = lambda x: x
    self.checkpoint_manager = checkpoint_manager.CheckpointManager(
        root_directory=self.config.checkpoint_root_directory,
        options=self.config.checkpointing_options,
    )
    # print("ckpt options", self.config.checkpointing_options)
    self.metrics_logger = metrics_logger.MetricsLogger(
        self.config.metrics_logging_options,
        metric_prefix=self.config.metric_prefix,
    )
    self.is_managed_externally = False

    self._train_steps = 0  # represent # of times model has been updated
    self._iter_steps = 0  # represent # of times trainer has looped
    self._throttler = inflight_throttler.InflightThrottler(
        max_inflight=training_config.max_inflight_computations
    )
    self._mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN
    self._has_aux = has_aux
    self._pbar = None
    self._flops_measured: bool = False

    # self._train_steps = self.checkpoint_manager.maybe_restore(
    #     self.model, restore_only_lora_params=self._lora_enabled
    # )
    self._iter_steps = self._train_steps * self.config.get_with_default(
        "gradient_accumulation_steps", 1
    )

    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None
    self._prof = profiler.Profiler(
        initial_step=self._iter_steps,
        max_step=self.config.max_steps,
        profiler_options=self.config.profiler_options,
    )
    self._buffered_train_metrics: MetricsBuffer | None = None
    self._prev_buffered_train_metrics: MetricsBuffer | None = None
    self._buffered_eval_metrics: MetricsBuffer | None = None
    self.training_hooks = None
    self.data_hooks = None
    self._grad_mask = None
    self.rng = jax.random.key(0)
    self.eval_fn = eval_fn
    self.eval_ds_back = None
    self.moments = None
    self.meta = None
    self.moments_notset = True
    dim = model.config.embed_dim
    kwargs = {"L":512, "d_h":dim, "d_delta":dim,  "v_dim":dim, "dtype": jnp.bfloat16}
    # self.cache_train = subsel_utils.GradRepCache(mode="single", M=1024, topk=16, tau=1e-2, **kwargs)
    # self.cache_val = subsel_utils.GradRepCache(mode="single",  M=128, topk=8, tau=1e-2, **kwargs)
    self.cache_train = None
    self.cache_val = None

    # self.cache_train = subsel_utils.GradRepCache(mode="single", M=1024*4, topk=64, tau=1e-3, **kwargs)
    # self.cache_val = subsel_utils.GradRepCache(mode="single",  M=1024, topk=8, tau=1e-3, **kwargs)

    # self.cache_train = subsel_utils.GradRepCache(M=1024*10, d_h=dim, v_dim=dim, topk=64, tau=1e-1, dtype=jnp.float32)
    # self.cache_val = subsel_utils.GradRepCache(M=1024*4, d_h=dim, v_dim=dim, topk=16, tau=1e-1, dtype=jnp.float32)

  def _validate_config(self, training_config: TrainingConfig):
    if (
        training_config.gradient_accumulation_steps is not None
        and training_config.eval_every_n_steps
        % training_config.gradient_accumulation_steps
        != 0
    ):
      raise ValueError(
          "eval_every_n_steps must be divisible by gradient_accumulation_steps,"
          f" but got {training_config.eval_every_n_steps} and"
          f" {training_config.gradient_accumulation_steps}"
      )

  def with_training_hooks(self, training_hooks: hooks.TrainingHooks):
    self.training_hooks = training_hooks

  def with_data_hooks(self, data_hooks: hooks.DataHooks):
    self.data_hooks = data_hooks

  def clear_jit_cache(self):
    """Clears the JIT cache of the train and eval step functions.

    This function should be called when the trainer is being reused after
    overiding the training related states, for example, the loss function.
    """
    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None

  def with_loss_fn(
      self,
      loss_fn: Callable[
          Concatenate[nnx.Module, P], ArrayLike | Tuple[ArrayLike, Any]
      ],
      has_aux: bool = False,
  ):
    self.clear_jit_cache()
    self.loss_fn = loss_fn
    self.eval_loss_fn = loss_fn
    self._has_aux = has_aux
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], _ModelInputT]
  ):
    """Generates model input from training input.

    NB: output of this function will be passed to the loss function, so the args
    should match what loss function expects.

    Args:
      gen_model_input_fn: A function that generates model input from training
        input.

    Returns:
      PeftTrainer.
    """
    self.clear_jit_cache()
    self.gen_model_input_fn = gen_model_input_fn
    return self

  def _train_stepasdf(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any, step,
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    """Main body for one train step.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      inputs: The training input.

    Returns:
      The loss and auxiliary data if has_aux is True, otherwise the loss.
    """
    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )
    # out, grads = grad_fn(model, **inputs)
    out, grads = grad_fn(model, inputs, {"step": step})
    optimizer.update(grads)
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any, step, loss_mask
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    """Main body for one train step.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      inputs: The training input.

    Returns:
      The loss and auxiliary data if has_aux is True, otherwise the loss.
    """
    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )
    # out, grads = grad_fn(model, **inputs)
    out, grads = grad_fn(model, inputs, {"step": step, "loss_mask": loss_mask})
    optimizer.update(grads)
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None

  def _eval_step(
      self, model: nnx.Module, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    inputs = self.gen_model_input_fn(inputs)
    model.eval()
    # out = self.eval_loss_fn(model, **inputs)
    out = self.eval_loss_fn(model, inputs)
    model.train()
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None

  def create_train_step_fn(self) -> Callable[..., ArrayLike]:
    """Creates the train step function."""
    return self._train_step

  def create_eval_step_fn(self) -> Callable[..., ArrayLike]:
    """Creates the eval step function."""
    return self._eval_step

  def _shard_optimizer(self, mesh: shd.Mesh, optimizer) -> None:
    """Optimizer states should be sharded before calling the jit function.

    If not, the _train_step will be compiled 2 times.

    Args:
      mesh: The mesh used for sharding.
    """
    if mesh.empty:
      return
    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)

    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, optimizer_pspecs
    )
    nnx.update(optimizer, optimizer_sharded_state)

  def jit_train_and_eval_step(self, skip_jit: bool = False):
    """Creates and returns the train and eval step functions.

    This function will return the cached ones if available.

    Args:
      skip_jit: If True, the train and eval step functions will not be JITed.

    Returns:
      A tuple of train and eval step functions.
    """
    train_step = self.create_train_step_fn()
    eval_step = self.create_eval_step_fn()
    if skip_jit:
      return train_step, eval_step
    else:
      if self._jitted_train_step_fn is None:
        self._shard_optimizer(pxla.thread_resources.env.physical_mesh, self.optimizer)
        self._shard_optimizer(pxla.thread_resources.env.physical_mesh, self.optimizer_head)
        self._jitted_train_step_fn = nnx.jit(
            train_step, donate_argnames=("optimizer",)
        )
        self._jitted_eval_step_fn = nnx.jit(
            eval_step, donate_argnames=("model",)
        )
      return self._jitted_train_step_fn, self._jitted_eval_step_fn

  def _shard_input(self, input_data: TrainingInput) -> TrainingInput:
    """Shards the input data across the available devices.

    Args:
      input_data: The input data to be sharded, expected to be a TrainingInput
        dataclass.

    Returns:
      The sharded TrainingInput.
    """
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty:
      return input_data

    # Check if the input is already sharded with the target mesh to avoid
    # re-sharding.
    is_sharded = jax.tree.map(
        lambda x: isinstance(x, jax.Array)
        and hasattr(x, "sharding")
        and hasattr(x.sharding, "mesh")
        and x.sharding.mesh == mesh,
        input_data,
    )
    if all(jax.tree.leaves(is_sharded)):
      return input_data

    pspec = shd.PartitionSpec(*self.config.data_sharding_axis)

    with jax.transfer_guard("allow"):
      return jax.tree.map(
          lambda x: jax.make_array_from_process_local_data(
              sharding_utils.get_sharding(x, mesh=mesh, pspec=pspec), x
          ),
          input_data,
      )

  def _prepare_inputs(self, input_data: Any) -> Any:
    """Override this function for additional input preparation."""
    return input_data

  def _post_process_train_step(self, aux: Any) -> None:
    """Override this function for post processing aux data from train step."""
    pass

  def _post_process_eval_step(self, aux: Any) -> None:
    """Override this function for post processing aux data from eval step."""
    pass

  def _try_get_learning_rate(self) -> float | None:
    """Returns the learning rate from the optimizer state if available."""
    try:
      return self.optimizer.opt_state.inner_opt_state[1].hyperparams["learning_rate"].value
    except AttributeError:
      for chainpart in self.optimizer.opt_state:
        if isinstance(chainpart, optax.EmptyState):
          break
        if hasattr(chainpart, "hyperparams"):
          return chainpart.hyperparams["learning_rate"].value
      return None

  def _log_metrics(
      self,
      loss: ArrayLike = None,
      step: int | None = None,
      step_time_delta: float | None = None,
      additional_metrics: Dict[str, ArrayLike] | None = None,
  ):
    """Logs the metrics to the metrics logger and console."""
    if loss is not None:
      perplexity = np.exp(loss)
      self.metrics_logger.log("loss", loss, self._mode, step)
      self.metrics_logger.log("perplexity", perplexity, self._mode, step)
    learning_rate = self._try_get_learning_rate()
    if learning_rate is not None:
      self.metrics_logger.log(
          "learning_rate", jax.device_get(learning_rate), self._mode, step
      )
    if step_time_delta is not None:
      self.metrics_logger.log(
          "step_time_sec", step_time_delta, self._mode, step
      )
      self.metrics_logger.log(
          "steps_per_sec", 1.0 / (step_time_delta + 1e-9), self._mode, step
      )

    if self._mode == metrics_logger.Mode.TRAIN:
      logging.info(
          "Train step %d training loss: %f  - training perplexity: %f",
          step,
          loss,
          perplexity,
      )
    for k, v in (additional_metrics or {}).items():
      self.metrics_logger.log(k, v, self._mode, step)

  def _buffer_metrics(
      self,
      metrics_buffer: MetricsBuffer | None,
      loss: ArrayLike,
      step: int,
      step_time_delta: float = 0.0,
      additional_metrics: Dict[str, ArrayLike] = {},
  ) -> MetricsBuffer:
    """Buffers metrics for the current step."""
    loss = np.array(loss)
    if metrics_buffer is None:
      metrics_buffer = MetricsBuffer(
          step=step,
          losses=[loss],
          step_time_deltas=[step_time_delta],
      )
      if additional_metrics:
        for name, val in additional_metrics.items():
          metrics_buffer.additional_metrics[name] = [np.array(val)]
    else:
      assert metrics_buffer.step == step
      metrics_buffer.losses.append(loss)
      metrics_buffer.step_time_deltas.append(step_time_delta or 0.0)
      if additional_metrics:
        for name, val in additional_metrics.items():
          if name not in metrics_buffer.additional_metrics:
            metrics_buffer.additional_metrics[name] = [np.array(val)]
          else:
            metrics_buffer.additional_metrics[name].append(np.array(val))
    return metrics_buffer

  def _write_train_metrics(self):
    """Writes previous buffered train metrics."""
    if self._prev_buffered_train_metrics is None:
      # skip the first step so we can overlap I/O with next step.
      self._prev_buffered_train_metrics = self._buffered_train_metrics
      self._buffered_train_metrics = None
      return
    # increment the step by one for logging purpose, because train_step is not
    # incremented until the next model update.
    self._prev_buffered_train_metrics.step += 1
    self._write_metrics(self._prev_buffered_train_metrics)
    self._may_update_pbar(
        self._tqdm_train_metrics,
        step=self._prev_buffered_train_metrics.step,
        loss=self._prev_buffered_train_metrics.loss,
        step_time=self._prev_buffered_train_metrics.step_time_delta,
    )
    self._prev_buffered_train_metrics = self._buffered_train_metrics
    self._buffered_train_metrics = None

  def _write_metrics(self, metrics_buffer: MetricsBuffer):
    self._log_metrics(
        loss=metrics_buffer.loss,
        step=metrics_buffer.step,
        step_time_delta=metrics_buffer.step_time_delta,
        additional_metrics={
            k: np.mean(v)
            for k, (
                v,
            ) in metrics_buffer.additional_metrics.items()
        },
    )

  @contextlib.contextmanager
  def _switch_mode(self, mode: metrics_logger.Mode):
    original_mode = self._mode
    self._mode = mode
    try:
      yield
    finally:
      self._mode = original_mode

  @property
  def _tqdm_train_metrics(self) -> list[str]:
    return ["loss", "perplexity", "steps_per_sec", "learning_rate"]

  def _may_update_pbar(
      self,
      metrics: list[str],
      step: int | None = None,
      loss: ArrayLike | None = None,
      step_time: float | None = None,
  ):
    """Updates the progress bar with the given metrics if available."""
    if self._pbar is not None:
      self._pbar.update_metrics(metrics, self._mode, ndigits=3)
      self._pbar.update()

    if self.training_hooks and self._mode == metrics_logger.Mode.TRAIN:
      self.training_hooks.on_train_step_end(self, step, loss, step_time)

  def train(
      self,
      train_ds: Iterable[Any],
      num_train_sources: int ,
      eval_ds: Iterable[Any] | None = None,
      dev_ds: Iterable[Any] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Training loop."""
    micro_bs = self.fullConfig["batch_size"]
    subsel = self.fullConfig["subset_select"]["enabled"]
    mode = self.fullConfig["subset_select"]["mode"]
    chex.disable_asserts()
    
    self.meta = {
      "prev_utils": jnp.zeros((num_train_sources,), dtype=jnp.float32),
      "gain": jnp.zeros((num_train_sources,), dtype=jnp.float32)
    }
    cached_subsel = partial(subsel_utils.subset_select, self.model, num_train_sources, self.project, 
              self.optimizer_head, self.cache_train, self.cache_val, self.fullConfig)
    
    cached_grads = partial(subsel_utils.process_gradsv2, self.model, 
                self.project, self.optimizer_head,self.cache_train, self.cache_val)
    
    def get_train():
      if not subsel: return 1
      _ratio = self.fullConfig["subset_select"]["ratio"]
      if mode == "full": _ratio = 1
      _buffer = self.fullConfig["subset_select"]["buffer"]
      batch_to_buffer =int(_buffer*_ratio) 
      return batch_to_buffer
    
    batch_to_buffer = get_train()
    tosel = micro_bs*batch_to_buffer
    train_step, eval_step = self.jit_train_and_eval_step(skip_jit)
    # train_step = nnx.cached_partial(train_step, self.model, self.optimizer)
   
    # subset_select_jit = nnx.cached_partial(subsel_utils.subset_select, 
    #                                        self.model, self.project, self.optimizer_head, tosel, mode)
    # subset_select_jit = partial(subsel_utils.subset_select, ratio=tosel, mode=mode)
    if not skip_jit:
      logging.info(
          "Training with mesh: %s. Compiled train_step cache size: %s",
          pxla.thread_resources.env.physical_mesh,
          train_step.jitted_fn._cache_size(),  # pytype: disable=attribute-error,protected-access
      )

    # if eval_ds:
    #   self._run_eval(eval_ds, eval_step)

    if self.config.max_steps is not None and self._pbar is None:
      self._pbar = progress_bar.ProgressBar(
          metrics_logger=self.metrics_logger,
          initial_steps=self._train_steps,
          max_steps=self.config.max_steps,
          description=self.config.pbar_description,
      )

    if self.training_hooks:
      self.training_hooks.on_train_start(self)

    train_iterator = iter(train_ds)
    # ex = next(train_iterator)
    index = 0
    last_step_completion_time = time.perf_counter()
    step = -1
    with utils.time_measure("Train loop"):
      while True:
        step += 1
        self._prof.maybe_activate(self._iter_steps)
        with utils.time_measure("Train step"):
        # with jax.profiler.StepTraceAnnotation(
        #     "train", step_num=self._iter_steps
        # ):
          train_example = None
          if self.data_hooks:
            train_example = self.data_hooks.load_next_train_batch(self)
          else:
            try:
              train_example = next(train_iterator)
              if not self.is_managed_externally:
                # TODO(mridulsahu): Add support to restore the iterator state
                # instead of skipping the already trained examples.
                if index < self._iter_steps:
                  # Skip the examples that are already trained.
                  index += 1
                  continue
              index += 1
            except StopIteration:
              pass

          if train_example is None:
            break

          # Stop training if max_steps is reached.
          if (
              self.config.max_steps is not None
              and self._train_steps >= self.config.max_steps
          ):
            break
          self.rng, key = jax.random.split(self.rng)
          
          train_example = self._prepare_inputs(train_example)
          train_example = self._shard_input(train_example)
          train_example = self.gen_model_input_fn(train_example)

          if subsel and self.moments_notset and mode not in ["full", "random"]:
            self.moments_notset = False
            dummy = jax.tree.map(lambda x: jnp.zeros_like(x), train_example)
            # dummy_grads, _, _ = subsel_utils._chunk_per_ex_grads(
            #     self.model, self.fullConfig["task_config"]["config"], key, dummy
            # )
            dummy_grads, _ = subsel_utils.per_ex_grads(self.model, step, train_example, self.fullConfig["task_config"]["config"], key, moments=None)
            dummy_grads = jax.tree.map(lambda x: jnp.zeros_like(x.mean(0)), dummy_grads)
            # jax.debug.print("dummy {}", jax.tree.map(lambda x: x.shape, dummy_grads))
            # raise
            self.moments = {
              "train": (dummy_grads, dummy_grads),
              "val": (dummy_grads, dummy_grads)
            }

          # if not self._flops_measured and not skip_jit:
          #   self._flops_measured = True

          #   tflops_per_step = system_metrics_calculator.measure_tflops_per_step(
          #       train_step_fn=train_step,
          #       model=self.model,
          #       optimizer=self.optimizer,
          #       train_example=train_example,
          #   )
          #   if tflops_per_step is not None:
          #     self.metrics_logger.log(
          #         "tflops_per_step", tflops_per_step, self._mode, 0
          #     )
          # graphdef, state = nnx.split((self.model, self.project, self.optimizer_head))
          val_batch = None
          if dev_ds is not None:
            anchorbs = self.fullConfig["task_config"]["config"]["subsel"]["anchorbs"]
            val_batch = self.val_sample(dev_ds, 100*step + 42, anchorbs, n_val=-1)
          lr = self.schedule(self._train_steps)
          
          # mega_batch, moments, subsel_aux = subsel_utils.subset_select(self.fullConfig, self.model, self.project, 
          #                                                   self.project2, 
          #                                                    self.optimizer, self.optimizer_last, self.optimizer_head,self.optimizer_head2, self.moments, 
          #                                                    step, tosel, mode, train_example, val_batch, lr, key, self.cache_train, self.cache_val)
          # with utils.time_measure("Gradient processing"):
          subsel_aux = {}
          #   grads, anchors, subsel_aux = cached_grads(train_example, val_batch, step, self.fullConfig["task_config"]["config"], key)
          # from jax.sharding import SingleDeviceSharding
          # device0 = jax.devices()[0]
          # grads = jax.device_put(grads,SingleDeviceSharding(device0))
          # anchors = jax.device_put(anchors,SingleDeviceSharding(device0))
          with utils.time_measure("Subsel", suppress_logging=True):
            mega_batch, moments, loss_mask, subsel_aux, meta = cached_subsel( self.moments, step, tosel, mode, train_example, val_batch, lr, key, self.meta)
          
          # with utils.time_measure("Subsel"):
          #   train_loss, aux = subsel_utils._train_step(self.model, self.project, self.optimizer, self.optimizer_head, self.cache_train, 
          #         self.cache_val, self.fullConfig, self.moments, step, tosel, mode, train_example, val_batch, lr, key)
          self.moments = moments
          self.meta = meta

          for i in range(batch_to_buffer):
          # for i in range(1):
            # i = 0
            train_example = jax.tree.map(lambda x: x[i*micro_bs: (i+1)*micro_bs], mega_batch)
            # train_example = mega_batch
            self._throttler.wait_for_next()
            if self.training_hooks:
              self.training_hooks.on_train_step_start(self)
            with utils.time_measure("ModelStep", suppress_logging=True):
              train_loss, aux = train_step(self.model, self.optimizer, train_example, self._train_steps, loss_mask)

            current_time = time.perf_counter()
            step_time_delta = current_time - last_step_completion_time
            last_step_completion_time = current_time

            self._throttler.add_computation(train_loss)
            self._buffered_train_metrics = self._buffer_metrics(
                self._buffered_train_metrics,
                loss=train_loss,
                step=self._train_steps,
                step_time_delta=step_time_delta,
                additional_metrics={**aux, **subsel_aux} if i == 0 else aux
            )
            # NB: put this after self._buffer_metrics is important
            self._post_process_train_step(aux)
            self._iter_steps += 1

            if (
                self._iter_steps
                % self.config.get_with_default("gradient_accumulation_steps", 1)
                == 0
            ):
              self._train_steps += 1
              self._write_train_metrics()

              # Checkpoint frequency is configured by checkpointing_options.
              self.checkpoint_manager.save(
                  self._train_steps,
                  self.model,
                  save_only_lora_params=self._lora_enabled,
              )
              to_eval = self._train_steps % self.config.eval_every_n_steps == 0

              if (eval_ds and to_eval):
                seed = step*100 + 42
                # seed = None
                self._run_eval(eval_ds, eval_step, seed=seed, k=self.fullConfig["task_config"]["config"]["evalbs"])
                
              if (self.eval_fn and to_eval):
                self.eval_fn(self.model, self._train_steps, self.metrics_logger)

        self._prof.maybe_deactivate(self._iter_steps)

    self._throttler.wait_for_all()
    if self.training_hooks:
      self.training_hooks.on_train_end(self)
    if not self.is_managed_externally:
      self.close()

  def _save_last_checkpoint(self):
    last_saved_step = self.checkpoint_manager.latest_step()
    if last_saved_step is None or last_saved_step < self._train_steps:
      self.checkpoint_manager.save(
          self._train_steps,
          self.model,
          save_only_lora_params=self._lora_enabled,
          force=True,
      )

  @property
  def train_steps(self) -> int:
    """Returns the number of train steps taken."""
    return self._train_steps

  @property
  def iter_steps(self) -> int:
    """Returns the number of iterator steps taken."""
    return self._iter_steps

  def close(self):
    """Closes the trainer and its associated resources.

    This includes writing any buffered metrics, saving the last checkpoint,
    and closing the checkpoint manager and metrics logger.
    """
    self._write_train_metrics()
    self._save_last_checkpoint()
    self.checkpoint_manager.close()
    self.metrics_logger.close()
    if self._pbar is not None:
      self._pbar.close()
      self._pbar = None

  # def val_sample(self, iterable, seed: int, batch_size: int):
  #   import random
  #   rng = random.Random(seed)

  #   # cache individual examples: list[dict[str, jax.Array]]
  #   if self.eval_ds_back is None:
  #     cache = []
  #     for batch in iterable:  # dict of arrays (B, ...)
  #       B = batch[next(iter(batch))].shape[0]
  #       cache.extend([{k: v[i] for k, v in batch.items() if k != "meta"} for i in range(B)])
  #     rng.shuffle(cache)
  #     # if n_val > 0: cache = cache[:n_val]
  #     self.eval_ds_back = cache

  #   n = len(self.eval_ds_back)
  #   idxs = rng.sample(range(n), k=min(batch_size, n))  # no replacement
  #   exs = [self.eval_ds_back[i] for i in idxs]

  #   eval_ex = {k: jnp.stack([ex[k] for ex in exs], axis=0) for k in exs[0]}
  #   eval_ex = self._prepare_inputs(eval_ex)
  #   eval_ex = self._shard_input(eval_ex)
  #   eval_ex = self.gen_model_input_fn(eval_ex)
  #   return eval_ex

  def val_sample(self, iterable, seed: int, batch_size: int, n_val: int = -1):
      import random
      rng = random.Random(seed)

      if self.eval_ds_back is None:
          cache = []
          for batch in iterable:
              # batch:
              # {
              #   "input_ids": [B, seq],
              #   "attention_mask": [B, seq, seq],
              #   "meta": {"sources": [B]}
              # }
              # Pick any non-meta key to get batch size
              first_key = next(k for k in batch.keys() if k != "meta")
              B = batch[first_key].shape[0]

              for i in range(B):
                  ex = {}
                  for k, v in batch.items():
                      if k == "meta":
                          ex["meta"] = {mk: mv[i] for mk, mv in v.items()}
                      else:
                          ex[k] = v[i]
                  cache.append(ex)

          rng.shuffle(cache)
          if n_val > 0: cache = cache[:n_val]
          self.eval_ds_back = cache

      # Sample a batch from the cached examples
      n = len(self.eval_ds_back)
      if n == 0:
          raise ValueError("eval_ds_back is empty; iterable produced no examples.")

      idxs = rng.sample(range(n), k=min(batch_size, n))
      exs = [self.eval_ds_back[i] for i in idxs]
      example0 = exs[0]

      eval_ex = {}
      for k, v0 in example0.items():
          if k == "meta":
              eval_ex["meta"] = {
                  mk: jnp.stack([ex["meta"][mk] for ex in exs], axis=0)
                  for mk in v0.keys()
              }
          else:
              eval_ex[k] = jnp.stack([ex[k] for ex in exs], axis=0)

      eval_ex = self._prepare_inputs(eval_ex)
      eval_ex = self._shard_input(eval_ex)
      eval_ex = self.gen_model_input_fn(eval_ex)
      return eval_ex

      
  def _reservoir_sample(self, iterable: Iterable[Any], k: int, rng) -> list[Any]:
    """Returns k random elements from an iterable in one pass (reservoir sampling)."""
    sample: list[Any] = []
    for i, x in enumerate(iterable):
      if i < k:
        sample.append(x)
      else:
        j = rng.randint(0, i)
        if j < k:
          sample[j] = x
    return sample

  def _run_eval(
      self,
      eval_ds: Iterable[Any],
      eval_step_fn: Callable[..., Any],
      seed=None,
      k: int = 8,
  ) -> None:
    """Runs evaluation loop."""
    logging.info("Running evaluation on train step %d.", self._train_steps)
    # --- NEW: sample k items from the iterable (fresh every call) ---
    if k is not None and k > 0:
      # if you want deterministic per-step, e.g. seed=self._train_steps
      # Note: this consumes eval_ds once to choose the subset, but we only *evaluate* k items
      import random
      rng = random.Random(seed)
      eval_batches = self._reservoir_sample(eval_ds, k, rng)
      eval_iterator = iter(eval_batches)
    else:
      eval_iterator = iter(eval_ds)
    step = -1
    with self._switch_mode(metrics_logger.Mode.EVAL):
      eval_loss, eval_steps = 0, 0
      while True:
        step += 1
        if self.data_hooks:
          eval_example = self.data_hooks.load_next_eval_batch(self)
        else:
          try:
            eval_example = next(eval_iterator)
          except StopIteration:
            eval_example = None
        if eval_example is None:
          break
        eval_example = self._prepare_inputs(eval_example)
        eval_example = self._shard_input(eval_example)
        if self.training_hooks:
          self.training_hooks.on_eval_step_start(self)
        loss, aux = eval_step_fn(self.model, eval_example)
        loss = jax.lax.stop_gradient(loss)
        self._buffered_eval_metrics = self._buffer_metrics(
            self._buffered_eval_metrics,
            loss=loss,
            step=self._train_steps,
        )
        self._post_process_eval_step(aux)
        eval_loss += loss
        eval_steps += 1

      if eval_steps == 0:
        logging.warning(
            "No eval examples found. Skipping eval metrics logging."
        )
        return

      self._write_metrics(self._buffered_eval_metrics)
      logging.info(
          "Train step %d eval loss: %f - eval perplexity: %f",
          self._train_steps,
          self.metrics_logger.get_metric("loss", "eval"),
          self.metrics_logger.get_metric("perplexity", "eval"),
      )
      self._buffered_eval_metrics = None
      if self.training_hooks:
        self.training_hooks.on_eval_step_end(self, eval_loss)


'''
adam
bs128
llama 3b
eval
100 iters
1e-2 20 iters
sim mat

'''