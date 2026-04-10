"""Custom Optax optimizers used by this repository.

These are lightweight JAX/Optax ports of ASGO and DASGO update rules.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax


ScalarOrSchedule = float | jax.Array | Callable[[jax.Array], jax.Array]


def _as_schedule_fn(learning_rate: ScalarOrSchedule) -> Callable[[jax.Array], jax.Array]:
  if callable(learning_rate):
    return learning_rate
  return lambda _: jnp.asarray(learning_rate, dtype=jnp.float32)


def _matrix_inv_sqrt(mat: jnp.ndarray, eps: float) -> jnp.ndarray:
  dim = mat.shape[0]
  eye = jnp.eye(dim, dtype=mat.dtype)
  mat = mat + eps * eye
  eigvals, eigvecs = jnp.linalg.eigh(mat)
  inv_sqrt_vals = jnp.power(jnp.maximum(eigvals, eps), -0.5)
  return (eigvecs * inv_sqrt_vals[None, :]) @ eigvecs.T


def _to_matrix(x: jnp.ndarray) -> tuple[jnp.ndarray, tuple[int, ...]]:
  if x.ndim == 2:
    return x, x.shape
  return x.reshape((x.shape[0], -1)), x.shape


class _AsgoPerParamState(NamedTuple):
  momentum: jnp.ndarray
  precond: jnp.ndarray


class AsgoState(NamedTuple):
  count: jax.Array
  per_param: Any


def asgo(
    learning_rate: ScalarOrSchedule,
    momentum: float = 0.9,
    beta2: float = 0.8,
    eps: float = 1e-10,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
  """ASGO optimizer as an Optax gradient transformation."""
  lr_fn = _as_schedule_fn(learning_rate)

  def init_fn(params):
    def _init_one(p: jnp.ndarray) -> _AsgoPerParamState:
      if p.ndim == 2:
        dim = min(p.shape[0], p.shape[1])
        precond = jnp.zeros((dim, dim), dtype=p.dtype)
      else:
        precond = jnp.zeros((1, 1), dtype=p.dtype)
      return _AsgoPerParamState(momentum=jnp.zeros_like(p), precond=precond)

    per_param = jax.tree.map(_init_one, params)
    return AsgoState(count=jnp.zeros([], dtype=jnp.int32), per_param=per_param)

  def update_fn(updates, state, params=None):
    if params is None:
      raise ValueError("ASGO requires current params for weight decay.")

    lr = jnp.asarray(lr_fn(state.count), dtype=jnp.float32)

    def _update_one(g, p, s: _AsgoPerParamState):
      if g.ndim < 2:
        raise ValueError("ASGO expects ndim >= 2 parameters; use AdamW for 1D params.")
      if g.ndim != 2:
        raise ValueError("Missing Training Param")

      next_m = momentum * s.momentum + (1.0 - momentum) * g
      rows, cols = g.shape
      if rows < cols:
        gram = g @ g.T
      else:
        gram = g.T @ g

      precond = beta2 * s.precond + (1.0 - beta2) * gram
      inv_precond = _matrix_inv_sqrt(precond, eps)
      eye = jnp.eye(precond.shape[0], dtype=precond.dtype)
      bad = ~jnp.all(jnp.isfinite(inv_precond))
      inv_precond = jnp.where(bad, eye, inv_precond)

      update = jax.lax.cond(
          rows < cols,
          lambda _: inv_precond @ next_m,
          lambda _: next_m @ inv_precond,
          operand=None,
      )
      norm = jnp.linalg.norm(update, ord="fro")
      scale = (0.2 * jnp.sqrt(rows * cols)) / (norm + 1e-12)
      update = update * scale

      # Match reference: decoupled weight decay followed by gradient step.
      delta = -lr * update - (lr * weight_decay) * p
      next_state = _AsgoPerParamState(momentum=next_m, precond=precond)
      return delta, next_state

    new_updates, new_states = jax.tree.map(
        _update_one, updates, params, state.per_param
    )
    return new_updates, AsgoState(count=optax.safe_int32_increment(state.count), per_param=new_states)

  return optax.GradientTransformation(init_fn, update_fn)


class _DasgoPerParamState(NamedTuple):
  momentum: jnp.ndarray
  precond: jnp.ndarray


class DasgoState(NamedTuple):
  count: jax.Array
  per_param: Any


def dasgo(
    learning_rate: ScalarOrSchedule,
    momentum: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
  """DASGO optimizer as an Optax gradient transformation."""
  lr_fn = _as_schedule_fn(learning_rate)

  def init_fn(params):
    def _init_one(p: jnp.ndarray) -> _DasgoPerParamState:
      if p.ndim >= 2:
        precond = jnp.zeros((p.shape[1],), dtype=p.dtype)
      else:
        precond = jnp.zeros((1,), dtype=p.dtype)
      return _DasgoPerParamState(momentum=jnp.zeros_like(p), precond=precond)

    per_param = jax.tree.map(_init_one, params)
    return DasgoState(count=jnp.zeros([], dtype=jnp.int32), per_param=per_param)

  def update_fn(updates, state, params=None):
    if params is None:
      raise ValueError("DASGO requires current params for weight decay.")

    lr = jnp.asarray(lr_fn(state.count), dtype=jnp.float32)

    def _update_one(g, p, s: _DasgoPerParamState):
      if g.ndim < 2:
        raise ValueError("DASGO expects ndim >= 2 parameters; use AdamW for 1D params.")
      if g.ndim != 2:
        raise ValueError("DASGO reference rule is implemented for 2D parameters only.")

      next_m = momentum * s.momentum + (1.0 - momentum) * g
      precond = beta2 * s.precond + (1.0 - beta2) * jnp.sum(g * g, axis=0)

      update = next_m * jnp.power(precond + eps, -0.5)
      norm = jnp.linalg.norm(update, ord="fro")
      scale = (0.2 * jnp.sqrt(g.shape[0] * g.shape[1])) / (norm + 1e-12)
      update = update * scale

      # Match reference: decoupled weight decay followed by gradient step.
      delta = -lr * update - (lr * weight_decay) * p
      next_state = _DasgoPerParamState(momentum=next_m, precond=precond)
      return delta, next_state

    new_updates, new_states = jax.tree.map(
        _update_one, updates, params, state.per_param
    )
    return new_updates, DasgoState(count=optax.safe_int32_increment(state.count), per_param=new_states)

  return optax.GradientTransformation(init_fn, update_fn)


class _ShampooPerParamState(NamedTuple):
  momentum: jnp.ndarray
  v_sq: jnp.ndarray
  preconds: tuple[jnp.ndarray, ...]
  inv_preconds: tuple[jnp.ndarray, ...]


class ShampooState(NamedTuple):
  count: jax.Array
  per_param: Any


def _matrix_inv_root_newton(
    mat: jnp.ndarray,
    eps: float,
    root: int,
    n_iter: int = 50,
    tolerance: float = 1e-6,
) -> jnp.ndarray:
  dim = mat.shape[0]
  eye = jnp.eye(dim, dtype=mat.dtype)
  alpha = -1.0 / float(root)

  mat_ridge = mat + eps * eye
  mat_norm = jnp.linalg.norm(mat_ridge, ord="fro")
  z = (root + 1.0) / (2.0 * (mat_norm + 1e-12))
  x0 = jnp.power(z, -alpha) * eye
  m0 = z * mat_ridge
  err0 = jnp.linalg.norm(m0 - eye, ord="fro") / (jnp.linalg.norm(m0, ord="fro") + 1e-12)

  def cond_fn(carry):
    i, _, _, err = carry
    return jnp.logical_and(i < n_iter, err > tolerance)

  def body_fn(carry):
    i, x, m, _ = carry
    m_p = alpha * m + (1.0 - alpha) * eye
    x = x @ m_p
    m = jnp.linalg.matrix_power(m_p, root) @ m
    err = jnp.linalg.norm(m - eye, ord="fro") / (jnp.linalg.norm(m, ord="fro") + 1e-12)
    return i + 1, x, m, err

  _, x, _, _ = jax.lax.while_loop(cond_fn, body_fn, (0, x0, m0, err0))
  bad = ~jnp.all(jnp.isfinite(x))
  return jnp.where(bad, eye, x)


def _mode_n_product(x: jnp.ndarray, mat: jnp.ndarray, axis: int) -> jnp.ndarray:
  # Multiply tensor x by square matrix mat along a specific mode/axis.
  out = jnp.tensordot(mat, x, axes=[[1], [axis]])
  return jnp.moveaxis(out, 0, axis)


def shampoo(
    learning_rate: ScalarOrSchedule,
    momentum: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-10,
    weight_decay: float = 0.0,
    update_freq: int = 1,
    inverse_order: int = 4,
) -> optax.GradientTransformation:
  """Shampoo-style optimizer as an Optax gradient transformation."""
  lr_fn = _as_schedule_fn(learning_rate)

  def init_fn(params):
    def _init_one(p: jnp.ndarray) -> _ShampooPerParamState:
      if p.ndim < 2:
        preconds = (jnp.zeros((1, 1), dtype=p.dtype),)
        inv_preconds = (jnp.zeros((1, 1), dtype=p.dtype),)
      else:
        preconds = tuple(jnp.zeros((d, d), dtype=p.dtype) for d in p.shape)
        inv_preconds = tuple(jnp.zeros((d, d), dtype=p.dtype) for d in p.shape)
      return _ShampooPerParamState(
          momentum=jnp.zeros_like(p),
          v_sq=jnp.zeros_like(p),
          preconds=preconds,
          inv_preconds=inv_preconds,
      )

    per_param = jax.tree.map(_init_one, params)
    return ShampooState(count=jnp.zeros([], dtype=jnp.int32), per_param=per_param)

  def update_fn(updates, state, params=None):
    if params is None:
      raise ValueError("Shampoo requires current params for weight decay.")

    lr = jnp.asarray(lr_fn(state.count), dtype=jnp.float32)
    step_num = state.count + 1

    def _update_one(g, p, s: _ShampooPerParamState):
      if g.ndim < 2:
        raise ValueError("Shampoo expects ndim >= 2 parameters; use AdamW for 1D params.")

      next_m = momentum * s.momentum + (1.0 - momentum) * g
      next_v = beta2 * s.v_sq + (1.0 - beta2) * (g * g)

      upd = next_m
      new_preconds = []
      new_inv_preconds = []
      for dim_id, dim in enumerate(g.shape):
        g_mode = jnp.moveaxis(g, dim_id, 0).reshape((dim, -1))
        precond = beta2 * s.preconds[dim_id] + (1.0 - beta2) * (g_mode @ g_mode.T)

        recompute = (step_num % update_freq) == 0
        inv_from_precond = _matrix_inv_root_newton(
          precond, eps=eps, root=inverse_order
        )
        inv_precond = jnp.where(recompute, inv_from_precond, s.inv_preconds[dim_id])

        upd = _mode_n_product(upd, inv_precond, dim_id)
        new_preconds.append(precond)
        new_inv_preconds.append(inv_precond)

      adam_update = next_m * jax.lax.rsqrt(next_v + eps)
      upd_norm = jnp.linalg.norm(upd, ord="fro")
      adam_norm = jnp.linalg.norm(adam_update, ord="fro")
      upd = upd * (adam_norm / (upd_norm + 1e-12))

      delta = -lr * upd - (lr * weight_decay) * p
      next_state = _ShampooPerParamState(
          momentum=next_m,
          v_sq=next_v,
          preconds=tuple(new_preconds),
          inv_preconds=tuple(new_inv_preconds),
      )
      return delta, next_state

    new_updates, new_states = jax.tree.map(
        _update_one, updates, params, state.per_param
    )
    return new_updates, ShampooState(count=optax.safe_int32_increment(state.count), per_param=new_states)

  return optax.GradientTransformation(init_fn, update_fn)
