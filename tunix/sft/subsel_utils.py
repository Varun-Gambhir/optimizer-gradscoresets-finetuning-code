from __future__ import annotations

from typing import Any, Tuple, NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from functools import partial
import optax
from tunix.sft import utils


def _partition_accounting(selected_mask, part_id, budget):
    """
    used[p] = how many elements already selected in partition p
    rem[p]  = remaining capacity in partition p (budget - used)
    cnt[p]  = how many total elements exist in partition p
    """
    P = budget.shape[0]

    # bincount over part_id; selected_mask acts as 0/1 weights
    used = jnp.bincount(
        part_id,
        weights=selected_mask.astype(jnp.int32),
        length=P,
    ).astype(budget.dtype)

    rem = budget - used

    cnt = jnp.bincount(part_id, length=P).astype(budget.dtype)
    return used, rem, cnt



def _sample_uniform_from_mask(key, pick_mask):
    """
    Uniformly sample an index from the True positions of pick_mask.

    Implemented as categorical over logits:
      logits[i] = 0     if allowed
               = -inf  otherwise
    This makes the distribution uniform over allowed indices.

    If pick_mask is empty:
      did_pick = False and we return dummy index 0 (caller must gate on did_pick).
    """
    did_pick = jnp.any(pick_mask)
    logits = jnp.where(pick_mask, 0.0, -jnp.inf)

    key, subkey = jax.random.split(key)
    picked = jax.lax.cond(
        did_pick,
        lambda k: jax.random.categorical(k, logits, axis=0).astype(jnp.int32),
        lambda k: jnp.int32(0),
        subkey,
    )
    return key, did_pick, picked


def test_shortlist(part_id, feasible, rem, part_s, feas_s, cap_s, short_s, sort_idx):
  P = rem.shape[0]
  # part_id range and feasible implies cap>0
  bad_pid = (part_id < 0) | (part_id >= P)
  jax.lax.cond(
      jnp.any(bad_pid),
      lambda _: jax.debug.print(
          "SHORTLIST CHECK FAIL: part_id out of range. P={} part_id(min,max)=({}, {}) bad_mask={}",
          P, jnp.min(part_id), jnp.max(part_id), bad_pid.astype(jnp.int32)
      ),
      lambda _: None,
      operand=None,
  )

  bad_feas = feas_s & (cap_s <= 0)
  jax.lax.cond(
      jnp.any(bad_feas),
      lambda _: jax.debug.print(
          "SHORTLIST CHECK FAIL: feasible item with cap<=0 in sorted space. cap_s={} bad_feas_mask={}",
          cap_s, bad_feas.astype(jnp.int32)
      ),
      lambda _: None,
      operand=None,
  )

  # Per-partition counts in original space
  feas_per_part = jnp.bincount(part_id, weights=feasible.astype(jnp.int32), length=P).astype(jnp.int32)
  expected_per_part = jnp.minimum(rem.astype(jnp.int32), feas_per_part)
  expected = jnp.sum(expected_per_part)

  # Actual shortlist-per-part from scan result (sorted space)
  short_per_part = jnp.bincount(part_s, weights=short_s.astype(jnp.int32), length=P).astype(jnp.int32)
  actual = jnp.sum(short_s.astype(jnp.int32))

  # Main invariant + detailed dump when it fails
  jax.lax.cond(
      expected != actual,
      lambda _: jax.debug.print(
          "SHORTLIST BUG (sorted scan): expected={} actual={} | rem={} feas_per_part={} expected_per_part={} short_per_part={} | part_s={} feas_s={} cap_s={} short_s={} sort_idx={}",
          expected, actual,
          rem, feas_per_part, expected_per_part, short_per_part,
          part_s, feas_s.astype(jnp.int32), cap_s, short_s.astype(jnp.int32),
          sort_idx
      ),
      lambda _: None,
      operand=None,
  )

  # Helpful always-on summary (comment out if too noisy)
  # jax.debug.print(
  #     "SHORTLIST DBG: #feasible={} #shortlisted={} has_pos={}",
  #     jnp.sum(feasible.astype(jnp.int32)),
  #     actual,
  #     jnp.any(feasible & (proxy > 0.0)),
  # )

@partial(jax.jit, static_argnames=("scores_relu"))
def _shortlist_sorted(part_id, rem, proxy, feasible, scores_relu):
    """
    shortlist builder in *sorted space*.

    What it does:
      1) score[i] = ReLU(proxy[i])
      2) sort items by (partition asc, score desc) using two stable argsorts
      3) scan left-to-right; for each partition p keep the first rem[p] feasible items

    Returns:
      sort_idx : [N]  (sorted position -> original index)
      part_s   : [N]  part_id in sorted order
      feas_s   : [N]  feasible in sorted order
      short_s  : [N]  shortlist mask in sorted order
      has_pos  : bool any feasible proxy>0 in original space
      scores   : [N]  ReLU(proxy) in original order (for logging/inspection)
    """
    N = part_id.shape[0]
    P = rem.shape[0]

    scores = jnp.where(scores_relu, jnp.maximum(proxy, 0.0), proxy)
    # scores = jnp.maximum(proxy, 0.0)

    NEG = jnp.array(-1e30, dtype=scores.dtype)
    key_score = jnp.where(feasible, scores, NEG)

    # Two-pass stable sort: score desc, then partition asc (stable)
    idx_score = jnp.argsort(-key_score, stable=True)
    idx_part  = jnp.argsort(part_id[idx_score], stable=True)
    sort_idx  = idx_score[idx_part]

    part_s = part_id[sort_idx]
    feas_s = feasible[sort_idx].astype(jnp.bool_)
    cap_s  = rem[part_s].astype(jnp.int32)

    def scan_body(counts, x):
        p, feas, cap = x
        c = counts[p]
        take = feas & (c < cap)
        counts = counts.at[p].add(take.astype(jnp.int32))
        return counts, take

    counts0 = jnp.zeros((P,), dtype=jnp.int32)
    counts_f, short_s = jax.lax.scan(
        scan_body,
        counts0,
        (part_s.astype(jnp.int32), feas_s, cap_s),
        length=N,
    )


    has_pos = jnp.any(feasible & (proxy > 0.0))
    test_shortlist(part_id, feasible, rem, part_s, feas_s, cap_s, short_s, sort_idx)

    dbg = partial(_dbg, False)
    dbg("scores {}", scores)
    dbg("has_pos {}", has_pos)
    dbg("sort_idx {}", sort_idx)
    dbg("part_s {}", part_s)
    dbg("feas_s {}", feas_s.astype(jnp.int32))
    dbg("short_s {}", short_s.astype(jnp.int32))
    dbg("cap_s {}", cap_s)

    # Fallback logic
    use_fallback = (~has_pos) & jnp.any(feasible)
    use_fallback = jnp.where(scores_relu, use_fallback, False)

    dbg(
        "fallback decision: has_pos={} any_feasible={} -> use_fallback={}",
        has_pos,
        jnp.any(feasible),
        use_fallback,
    )

    pick_s = jnp.where(use_fallback, feas_s, short_s)

    dbg("pick_s (sorted mask) {}", pick_s.astype(jnp.int32))
    dbg("pick_s count {}", jnp.sum(pick_s.astype(jnp.int32)))


    return sort_idx, pick_s, use_fallback

def _sample_uniform_from_mask_sorted(key, pick_mask_s):
    did_pick = jnp.any(pick_mask_s)
    logits = jnp.where(pick_mask_s, 0.0, -jnp.inf)
    key, subkey = jax.random.split(key)
    picked_s = jax.lax.cond(
        did_pick,
        lambda k: jax.random.categorical(k, logits, axis=0).astype(jnp.int32),
        lambda k: jnp.int32(0),
        subkey,
    )
    return key, did_pick, picked_s


def _dbg(enabled, fmt, *args):
    return jax.lax.cond(
        enabled,
        lambda _: jax.debug.print(fmt, *args),
        lambda _: None,
        operand=None,
    )


@jax.jit
def select_one_partition_matroid_toprelu(key, selected_mask, part_id, budget, proxy):
    dbg = partial(_dbg, False)

    # Partition accounting
    used, rem, cnt = _partition_accounting(selected_mask, part_id, budget)

    dbg("SELECT STEP -----------------------------")
    dbg("part_id {}", part_id)
    dbg("selected_mask {}", selected_mask.astype(jnp.int32))
    dbg("used {}", used)
    dbg("budget {}", budget)
    dbg("rem {}", rem)
    dbg("cnt {}", cnt)

    # Feasibility
    rem_i = rem[part_id]
    feasible = (~selected_mask) & (rem_i > 0)

    dbg("rem_i {}", rem_i)
    dbg("feasible {}", feasible.astype(jnp.int32))
    dbg(
        "feasible stats: #feasible={} #not_sel={} rem_i(min,max)=({}, {})",
        jnp.sum(feasible.astype(jnp.int32)),
        jnp.sum((~selected_mask).astype(jnp.int32)),
        jnp.min(rem_i),
        jnp.max(rem_i),
    )

    # Shortlist (sorted space)
    sort_idx, pick_s, use_fallback = _shortlist_sorted(
        part_id, rem, proxy, feasible, scores_relu=False
    )

    # Sampling (sorted space)
    key, did_pick, picked_s = _sample_uniform_from_mask_sorted(key, pick_s)
    picked = sort_idx[picked_s]

    dbg(
        "sampling: did_pick={} picked_s={} picked(orig)={}",
        did_pick,
        picked_s,
        picked,
    )

    dbg(
        "picked feasibility check: feasible[picked]={} rem[picked_part]={}",
        feasible[picked],
        rem[part_id[picked]],
    )

    # Apply selection
    new_selected_mask = jax.lax.cond(
        did_pick,
        lambda m: m.at[picked].set(True),
        lambda m: m,
        selected_mask,
    )

    picked_random = did_pick & use_fallback

    dbg(
        "FINAL: did_pick={} picked_random={} new_selected_count={}",
        did_pick,
        picked_random,
        jnp.sum(new_selected_mask.astype(jnp.int32)),
    )
    dbg("END SELECT STEP -------------------------")

    return key, new_selected_mask, did_pick, picked, picked_random


@partial(jax.jit, static_argnames=("max_iters"))
def joint_subsel(rng, part_id, budget, scores, apdg_lr, interaction_matrix, prev_utils, max_iters):
    rng, key = jax.random.split(rng)
    Bs = part_id.shape[0]
    weights = jnp.zeros((Bs,), dtype=jnp.float32)
    selected_mask = jnp.zeros((Bs,), dtype=jnp.bool_)
    iters = jnp.sum(budget)
    picked_idxs = -jnp.ones((Bs,), dtype=jnp.int32)
    
    refit_fn = partial(apgd_partitions, (scores, interaction_matrix), part_id, max_iters, apdg_lr)
    grad_fn = partial(grad_utility, (scores, interaction_matrix))
    utilities = -jnp.inf*jnp.ones((Bs,max_iters), dtype=jnp.float32)

    @jax.jit
    def omp_step(key, weights, selected_mask, part_id, budget):
        # weights = jnp.where(selected_mask, weights, 0.0)
        proxy = grad_fn(weights)  # [N]

        key, selected_mask, did_pick, picked, picked_random = select_one_partition_matroid_toprelu(
            key, selected_mask, part_id, budget, proxy
        )
        _util = -jnp.inf*jnp.ones((max_iters,), dtype=jnp.float32) 

        weights, utility = jax.lax.cond(
            did_pick,
            lambda w: refit_fn(selected_mask, key, random_init=True),
            lambda w: (w, _util),
            weights,
        )

        return key, weights, utility, selected_mask, did_pick, picked, picked_random

    def body(i, state):
        key, weights, utilities, selected_mask, n_random, n_opt, picked_idxs = state
        key, weights, utility, selected_mask, did_pick, picked, picked_random = omp_step(
            key, weights, selected_mask, part_id, budget
        )
        picked_idxs = picked_idxs.at[i].set(picked)
        utilities = utilities.at[i].set(utility)
        # jax.debug.print("iter {}:{} ###### \n ", i, did_pick)

        # Accumulate counts
        n_random = n_random + picked_random.astype(jnp.int32)
        n_opt = n_opt + (did_pick & (~picked_random)).astype(jnp.int32)

        return key, weights, utilities, selected_mask, n_random, n_opt, picked_idxs

    init_state = (key, weights, utilities, selected_mask, jnp.int32(0), jnp.int32(0), picked_idxs)
    _, weights, utilities, mask, n_random, n_opt, picked_idxs = jax.lax.fori_loop(0, iters, body, init_state)

    jax.debug.print("joint_subsel done: picked_random={} picked_opt={}", n_random, n_opt)

    # t = jax.tree.map(lambda x, y: (x,y), part_id, picked_idxs, is_leaf=)
    jax.debug.print("partition {}", part_id)
    jax.debug.print("selected order {}", picked_idxs)
    jax.debug.print("bs/iters {} {}", Bs, iters)
    return mask, utilities, weights


@partial(jax.jit, static_argnames=("random_init", "max_iterations"))
def apgd_partitions(
    A: jnp.ndarray,            # [n, n]
    part_id: jnp.ndarray,      # [n]
    max_iterations: int, 
    learning_rate: float,
    selected_mask: jnp.ndarray,# [n] bool
    rng,
    random_init=False
):
    def project(w):
      """
        project over support
        w >= 0
        w_i = 0 if selected_mask[i] == False
      """
      w = jnp.where(selected_mask, w, 0.0)
      return jnp.maximum(w, 0.0)
      # return w
    w_init = jnp.zeros(part_id.shape, dtype=jnp.float32)
    if random_init:
      w_init = jax.random.uniform(rng, shape=part_id.shape)
    w = project(w_init)
    utility = -jnp.inf*jnp.ones(max_iterations, dtype=jnp.float32)
    y = w
    t = 1.0

    def body(iter, state):
        w, y, t, utility = state
        g = grad_utility(A, y)

        # ASCENT
        w_next = project(y + learning_rate * g)

        t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        y_next = w_next + ((t - 1.0) / t_next) * (w_next - w)
        _util = utility_fn(A, w_next)
        # jax.debug.print("inside util {}", _util)
        utility_new = utility.at[iter].set(_util)

        return (w_next, y_next, t_next, utility_new)

    w, _, _, utility = jax.lax.fori_loop(
        0, max_iterations, body, (w, y, t, utility)
    )

    return w, utility

def utility_fn(A, w):
    '''
    utility for greats obejective wrt w
    scores: [bs_train]
    interaction: [bs_train, bs_train]
    w: [bs_train]
    '''
    scores, interaction_matrix = A
    Bs = scores.shape[0]
    interaction = w*(interaction_matrix@w)
    chex.assert_shape(interaction, (Bs,))
    utility = jnp.sum(scores*w) - 0.5*jnp.sum(interaction)
    # DONOT mask non support entries
    # since support of w need not != support of g
    return utility

def grad_utility(A, w):
    '''
    gradient for greats obejective wrt w
    '''
    scores, interaction_matrix = A
    # g = scores - interaction_matrix@w
    g = scores - (interaction_matrix@w)
    # DONOT mask non support entries
    # since support of w need not != support of g
    return g


@partial(nnx.jit, static_argnames=("out_len"))
def greats_selection(scores, interaction_matrix, out_len: int, source_mask=None, limit=None):
  """
  scores: (n,) 1D array
  interaction_matrix: (n, n)
  out_len: int
  returns: (K,) int32 selected indices
  """
  if limit is None: limit = out_len
  W = interaction_matrix
  if source_mask is not None:
    chex.assert_equal_shape(scores, source_mask)
    chex.assert_type(source_mask, jnp.bool)
    scores = jnp.where(source_mask, scores, -jnp.inf)

  effective_k = jnp.minimum(out_len, jnp.where(limit is None, out_len, limit))
  # Buffers (static shapes)
  selected0 = -jnp.ones((out_len,), dtype=jnp.int32)  # filled prefix; unused stay -1

  def body(i, state):
    cur_scores, selected = state
    idx = jnp.argmax(cur_scores)                      # int32
    selected = selected.at[i].set(idx)
    cur_scores = cur_scores - W[idx, :]  # subtract interactions
    cur_scores = cur_scores.at[idx].set(-jnp.inf)       # prevent reselection
    return (cur_scores, selected)

  final_scores, selected = jax.lax.fori_loop(0, effective_k, body, (scores, selected0))
  return selected  # [k], with first `effective_k` filled, rest = -1



@partial(jax.jit, static_argnums=(1,))
def facility_location_old2(S: jnp.ndarray, k: int) -> jnp.ndarray:
  """
  Args:
      S: [n, m] similarity matrix. Objective is sum_i max_{j in A} S[i, j].
      k: number of items to select (must be <= n).

  Returns:
      idxs: [k] int32 array of selected indices.
  """
  S = jnp.transpose(S) # (m,n) -> thus n is axis where points are choosen from
  m, n = S.shape
  # k = jnp.minimum(k, n)

  # best[i] = best similarity achieved so far for row i (coverage)
  # best = jnp.full((n,), -jnp.inf, dtype=S.dtype)
  # best = jnp.zeros((n,), dtype=S.dtype)
  # selected_mask = jnp.zeros((n,), dtype=bool)
  # out = jnp.full((k,), -1, dtype=jnp.int32)
  gains0 = S.sum(axis=0)
  j0 = jnp.argmax(gains0) # choosing axis = 1
  best = S[:, j0]   # best[i] = max_{j in {j0}} S[i, j]

  # bookkeeping
  selected_mask = jnp.zeros((n,), dtype=bool).at[j0].set(True)
  out = jnp.full((k,), -1, dtype=jnp.int32).at[0].set(j0)


  def step(t, carry):
    best, selected_mask, out = carry
    # Marginal gain for adding column j: sum_i max(0, S[i,j] - best[i])
    gains = jnp.maximum(S - best[:, None], 0.0).sum(axis=0)
    # Mask already-selected candidates
    gains = jnp.where(selected_mask, -jnp.inf, gains)

    j = jnp.argmax(gains)  # tie-breaks to smallest index
    # Update coverage and bookkeeping
    best = jnp.maximum(best, S[:, j])
    selected_mask = selected_mask.at[j].set(True)
    out = out.at[t].set(jnp.int32(j))
    return (best, selected_mask, out)

  best, selected_mask, out = jax.lax.fori_loop(1, k, step, (best, selected_mask, out))
  jax.debug.print("selected {}", out)
  return out

@partial(jax.jit, static_argnums=(1,))  # match  k, reg, iters static
def facility_location(
    S: jnp.ndarray,                 # [n, n] similarities
    k,                         # capacity of output array (static length)
    src_mask: jnp.ndarray = None,   # [n] bool: eligible columns (candidates)
    tgt_mask: jnp.ndarray = None,   # [n] bool: rows that contribute to coverage
    limit: jnp.ndarray | None = None,  # dynamic effective k for this call
) -> jnp.ndarray:
  # S = (1+S)/2
  """
  Facility-Location greedy with early stop and padding:
    - Returns int32 array of length k.
    - Fills only the first `effective_k` positions; the rest remain -1.
    - `effective_k = min(k, limit, #eligible)`; if no eligible, returns all -1.
    - Masks: src_mask filters candidate columns; tgt_mask filters covered rows.
  """
  n = S.shape[0]

  # default masks
  src_mask = jnp.ones((n,), dtype=bool) if src_mask is None else src_mask
  tgt_mask = jnp.ones((n,), dtype=bool) if tgt_mask is None else tgt_mask

  # bound effective steps by k, limit (if given), and #eligible candidates
  max_possible = jnp.sum(src_mask.astype(jnp.int32))
  base_k = k if limit is None else jnp.minimum(k, limit)
  effective_k = jnp.minimum(base_k, max_possible)

  # static-shaped buffers
  selected = -jnp.ones((k,), dtype=jnp.int32)
  chosen   = jnp.zeros((n,), dtype=bool)
  neg_inf = -jnp.inf
  S_clean = jnp.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
  # early exit: nothing to pick
  def early_return(_):
    return selected
  def proceed(_):
    # bootstrap: pick first column by masked column-sum over covered rows
    # gains0[j] = sum_{i in tgt_mask} S[i, j], masked by src eligibility
    gains0 = (S_clean * tgt_mask[:, None]).sum(axis=0)
    gains0 = jnp.where(src_mask, gains0, neg_inf)
    j0 = jnp.argmax(gains0)

    selected0 = selected.at[0].set(j0)
    chosen0   = chosen.at[j0].set(True)
    # coverage vector over rows; we’ll always mask rows via tgt_mask in deltas
    best0 = S[:, j0]

    # loop state
    state0 = (best0, chosen0, selected0, 1)

    def cond_fun(state):
      _, _, _, t = state
      return t < effective_k

    def body_fun(state):
      best, chosen, selected, t = state
      # marginal gains: sum_i max(0, S[i,j] - best[i]) only over tgt_mask rows
      delta = S_clean - best[:, None]
      delta = jnp.where(tgt_mask[:, None], delta, neg_inf)  # masked rows contribute 0 after relu
      delta = jnp.where(src_mask[None, :], delta, neg_inf)  # masked cols contribute 0 after relu
      gains = jnp.maximum(delta, 0.0).sum(axis=0)
      # block already chosen and ineligible columns
      valid = src_mask & (~chosen)
      gains = jnp.where(valid, gains, neg_inf)

      j = jnp.argmax(gains)
      # update coverage and bookkeeping
      best = jnp.maximum(best, S[:, j])
      best = jnp.where(tgt_mask, best, neg_inf)
      chosen  = chosen.at[j].set(True)
      selected = selected.at[t].set(jnp.int32(j))
      t = t + 1
      return (best, chosen, selected, t)

    _, _, selected_fin, _ = jax.lax.while_loop(cond_fun, body_fun, state0)
    return selected_fin

  # If effective_k == 0, return all -1; else run greedy.
  idx = jax.lax.cond(effective_k == 0, early_return, proceed, operand=None)
  # jax.debug.print("selected {} {}", effective_k, idx)
  return idx

def tree_sq_norm(pytree):
    # [B] per-sample squared L2 across all leaves
    sq = jax.tree.map(lambda x: jnp.sum(x.astype(jnp.float32)**2, axis=-1), pytree)
    return jax.tree.reduce(lambda a, b: a + b, sq)

def normalize(pytree, eps=1e-8):
    # in: pytree{[B,d_l]} → out: pytree{[B,d_l]} with global L2=1 per sample
    n2 = tree_sq_norm(pytree)
    inv = (1.0 / jnp.sqrt(n2 + eps))[:, None]  # [B,1]
    return jax.tree.map(lambda x: x * inv, pytree)

def gram_linear(X, Y=None):
    if Y is None: Y = X
    # return X@Y.T

    # sum_l X_l X_lᵀ → [B,B]
    def leaf_gram(x, y):
        x = x.astype(jnp.float32)              # [B,d]
        y = y.astype(jnp.float32)              # [B,d]
        return x @ y.T                         # [B,B]
    return jax.tree.reduce(lambda a, b: a + b, jax.tree.map(leaf_gram, X, Y))

def update_moments(ex_grads, mu, nu, count):
  # if not lowpass_adam: return ex_grads, None, None
  # if mu is None: return ex_grads, None, None
  eps=1e-8
  b1=0.9
  b2=0.999 

  def bias_correct(mu, nu, count):
    count_inc = count + 1
    mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
    nu_hat = optax.tree.bias_correction(nu, b2, count_inc)
    return mu_hat, nu_hat
  
  def get_mu_nu(grad, mu, nu):
    mu = optax.tree.update_moment(grad, mu, b1, 1)
    nu = optax.tree.update_moment_per_elem_norm(grad, nu, b2, 2)
    return mu, nu
    
  # jax.debug.print("grads shsape {}", jax.tree.map(lambda x: x.shape, ex_grads))
  def scale(ex_grad):
    _mu, _nu = get_mu_nu(ex_grad, mu, nu)
    _mu_hat, _nu_hat = bias_correct(_mu, _nu, count)
    smooth = jax.tree.map(
      lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps), _mu_hat, _nu_hat, is_leaf=lambda x: x is None,
    )

    return smooth, _mu, _nu
  
  grads, mu_batch, nu_batch = jax.vmap(scale)(ex_grads)
  # grads = ex_grads

  return grads, mu_batch, nu_batch

def grad_filter(grad_layer, path, value):
  # jax.debug.print("path {}",path)
  # ('layers', Array(1, dtype=int32), 'mlp', 'down_proj', 'kernel_lora_a')
  if len(path) > 1 and isinstance(path[1], (int, jax.Array)):
    # idx = int(path[1]) if isinstance(path[1], jax.Array) else path[1]
    idx = int(path[1]) if isinstance(path[1], jax.Array) else path[1]
    if idx in grad_layer:
      for cand in ["w_lora_a", "w_lora_b", "kernel_lora_a", "kernel_lora_b"]:
         if cand in path[-1]:
          return True
  return False



def cross_entropy_loss(logits, targets, temp=1.0):
  
  targets = jax.nn.softmax(targets/temp, axis=-1)
  log_probs = jax.nn.log_softmax(logits/temp, axis=-1)
  return -jnp.sum(targets * log_probs, axis=-1).mean()

def mse_loss(logits, targets):
  loss = (logits - targets)**2
  loss = loss.mean()
  return loss

def pool_hidden(hidden, mask):
  denom = jnp.maximum(mask.sum(1, keepdims=True), 1.0)
  hidden = (hidden * mask[..., None]).sum(1) / denom
  return hidden


def pack_pytree(pytree):
  # tree leaves: [(B, D1), (B, D2), ...] -> concat: (B, Dtot)
  leaves, _ = jax.tree_util.tree_flatten(pytree)
  X = jnp.concatenate(leaves, axis=1)          # (B, Dtot)
  return X

# @partial(nnx.jit, static_argnames=("Dtot"))
def pad_features(x, Dtot):
    B, Dsmall = x.shape
    # if Dsmall == Dtot: return x
    assert Dsmall <= Dtot
    pad_width = ((0, 0), (0, Dtot - Dsmall))
    return jnp.pad(x, pad_width, mode="constant", constant_values=0)

@partial(nnx.jit)
def model_out(model, inputs):
  loss, (hidden, delta) = per_ex_loss(
      model,
      inputs["input_tokens"],
      inputs["input_mask"],
      inputs["positions"],
      inputs["attention_mask"],
  )
  return loss, hidden.astype(jnp.bfloat16), delta.astype(jnp.bfloat16)

def wt_mean(task_cfg, max_num_sources, sources, grads):
    if not task_cfg["subsel"]["val_srcwt"]: return grads
    class_ids = jnp.arange(max_num_sources, dtype=jnp.int32) # [MC]
    src_mask = (sources[None, :] == class_ids[:, None])       # [MC, N]
    num_src = jnp.sum(jnp.diff(jnp.sort(sources)) != 0) + 1
    counts = src_mask.sum(1) # [MC]
    wt_src = jnp.where(counts > 0.0, 1/counts, 0.0) # [MC]
    wt_src = wt_src/num_src
    wt_ex = wt_src[sources] # [N] 
    # anch [N, d]
    grads = (grads*wt_ex[:, None])
    grads_anchor = grads.sum(0, keepdims=True) # [1, d]
    chex.assert_tree_shape(grads_anchor, (1, grads.shape[1]))
    jax.debug.print("wt_ex {} {} {} sum={}", counts, num_src, wt_ex, wt_ex.sum())
    return grads_anchor




@partial(
  nnx.jit, 
  static_argnames=("task_cfg"), 
  donate_argnames=("model", "project", "optimizer_head", "cache_train", "cache_val")
)
def process_gradsv2(model, project, optimizer_head, cache_train, 
                    cache_val, inputs_train, inputs_val, moments_train, moments_val,
                    step, task_cfg, rng):
  loss = {}
  ret_meta = {
     "moments_train": None,
     "moments_val": None,
  }
  lowpass = task_cfg["grads"]["lowpass_adam"]


  warmup = task_cfg["gradsapprox"]["gradapprox_warmup"]

  approx_mode = task_cfg["gradsapprox"]["grads_approx"]
  if approx_mode == "none":
    grads_train, meta_train = per_ex_grads(model, step, inputs_train, task_cfg, 
                                           rng, lowpass=lowpass,  moments=moments_train, chunk_size=16)
    if task_cfg["subsel"]["val_anchors"]:
      grads_val, meta_val = per_ex_grads(model, step, inputs_val, task_cfg, rng,
                                         lowpass=lowpass, moments=moments_val, chunk_size=16)
    else:
      grads_val, meta_val = grads_train, meta_train

    grads_hat_train, grads_hat_val = grads_train, grads_val
    ret_meta["moments_train"] = meta_train["moments"]
    ret_meta["moments_val"] = meta_val["moments"]
  
  else:
    raise ValueError

  if task_cfg["grads"]["normalize"] == True:
    grads_train = normalize(grads_train)
    grads_val = normalize(grads_val)

  return grads_train, grads_val, loss, ret_meta

def updatedict(tag, kv, mega):
   for k in kv.keys():
      mega[f"{k}_{tag}"] = kv[k]
   return mega

@partial(jax.jit, static_argnames=("ratio"))
def conflicting(grads, idxs, ratio):
  chex.assert_shape(idxs, (ratio,))
  # This is a hack, just to satisfy the compiler
  # idxs is already of shape [ratio]
  # as asserted above
  idxs = idxs[:ratio]
  grads = pack_pytree(grads)
  grads = grads[idxs]
  kernel = gram_linear(grads, grads)
  n = kernel.shape[0]
  kernel = kernel*(1 - jnp.eye(n,n))
  pairs = (jnp.sum(kernel < 0))/2
  # agree = (jnp.sum(kernel > 0) - n)/2
  return pairs

def _adjust_to_target(bud, cnt, target, key):
  # bud, cnt: int32 [C]
  # target: scalar int32
  init_diff = target - jnp.sum(bud)
  big = jnp.float32(1e9)

  def cond_fun(state):
    bud, diff, key = state
    return diff != 0

  def valid_idx(subkey, mask, bud):
    u = jax.random.uniform(subkey, bud.shape)
    u_masked = jnp.where(mask, u, big)
    idx = jnp.argmin(u_masked)
    return idx

  def body_fun(state):
    bud, diff, key = state
    key, subkey = jax.random.split(key)
    big = jnp.float32(1e9)

    def add_step(_):
      mask = bud < cnt  # can still increase
      any_valid = jnp.any(mask)

      def do_add(_):
        idx = valid_idx(subkey, mask, bud)
        bud2 = bud.at[idx].add(1)
        diff2 = diff - 1
        return bud2, diff2, key

      def no_add(_):
        # no valid positions; stop by zeroing diff
        return bud, jnp.int32(0), key

      return jax.lax.cond(any_valid, do_add, no_add, operand=None)

    def sub_step(_):
      mask = bud > 0  # can still decrease
      any_valid = jnp.any(mask)

      def do_sub(_):
        idx = valid_idx(subkey, mask, bud)
        bud2 = bud.at[idx].add(-1)
        diff2 = diff + 1
        return bud2, diff2, key

      def no_sub(_):
        return bud, jnp.int32(0), key

      return jax.lax.cond(any_valid, do_sub, no_sub, operand=None)

    return jax.lax.cond(diff > 0, add_step, sub_step, operand=None)

  init_state = (bud, init_diff.astype(jnp.int32), key)
  final_bud, final_diff, final_key = jax.lax.while_loop(cond_fun, body_fun, init_state)
  return final_bud


def standardize(x, axis=None):
  """
  Standardize tensor x along the given axis.
  If std == 0, outputs zeros on those entries instead of dividing.
  """
  mean = jnp.mean(x, axis=axis, keepdims=True)
  var = jnp.var(x, axis=axis, keepdims=True)
  std = jnp.sqrt(var)

  # Where std > 0, use (x-mean)/std; otherwise return 0
  standardized = jnp.where(std > 0, (x - mean) / std, 0.0)
  return standardized


def curr_budget(task_cfg, lamb, part_id, budget, gain, ratio, key):
  if not task_cfg["curricullum"]["enabled"]: return budget
  P = budget.shape[0]
  cnt = jnp.bincount(part_id, length=P).astype(jnp.int32)

  tot_gain = jnp.sum(gain)
  C = gain.shape[0]
  term = standardize(gain, axis=0)
  new_bud = ratio / C - (ratio / lamb) * term  # float [C]
  jax.debug.print("debug sum {}", jnp.sum(new_bud))
  # per-class bounds
  new_bud = jnp.clip(new_bud, 0.0, cnt.astype(new_bud.dtype))
  new_bud = jnp.rint(new_bud).astype(jnp.int32)
  adap_bud = new_bud

  # global target, clamped to capacity
  total_cap = jnp.sum(cnt)
  target = jnp.round(ratio).astype(jnp.int32)
  target = jnp.clip(target, 0, total_cap)

  # random, exact adjustment under jit
  new_bud = _adjust_to_target(new_bud, cnt, target, key)
  jax.debug.print("orig_bud {} {}", budget, jnp.sum(budget))
  jax.debug.print("adap_bud {} {}", adap_bud, jnp.sum(adap_bud))
  jax.debug.print("new_bud {} {}", new_bud, jnp.sum(new_bud))
  return new_bud


def per_class_utilities(grads, anchors, src_v, num_train_sources, lr1, lr2, interaction_matrix, weights):
    # sims: [N, A]
    sims = gram_linear(grads, anchors).astype(jnp.float32)

    # one_hot: [A, C]
    one_hot = jax.nn.one_hot(src_v, num_classes=num_train_sources, dtype=sims.dtype)

    # masked mean over anchors per class
    sims_exp = sims[:, :, None]           # [N, A, 1]
    mask = one_hot[None, :, :]           # [1, A, C]

    num = jnp.sum(sims_exp * mask, axis=1)      # [N, C]
    denom = jnp.sum(mask, axis=1)               # [1, C]
    denom = jnp.clip(denom, 1.0)                # avoid div by zero
    scores_per_class = num / denom              # [N, C]

    def util_for_class(scores_c):
        return utility_fn((lr1 * scores_c, lr2 * interaction_matrix), weights)

    # scores_per_class: [N, C], vmap over classes (axis=1)
    final_utils = jax.vmap(util_for_class, in_axes=1, out_axes=0)(scores_per_class)  # [C]
    return final_utils



@partial(
  nnx.jit, 
  static_argnames=("ratio", "mode", "config", "num_train_sources"),
  donate_argnames=("model", "project", "optimizer_head", "cache_train", "cache_val")
)
def subset_select(model, num_train_sources, project, optimizer_head, cache_train, 
                  cache_val, config, moments, step, ratio, mode, inputs, val_batch, lr, rng, cls_meta):
# def subset_select(config, model, project, project2, optimizer, optimizer_last, optimizer_head, optimizer_head2, 
#                   moments, step, ratio, mode, inputs, val_batch, lr, rng, cache_train, cache_val):
  # jax.debug.print("input kets befor{}e", inputs.keys())
  bs = jax.tree.leaves(inputs)[0].shape[0]
  task_cfg = config["task_config"]["config"]

  inputs_meta = None
  if "meta" in inputs.keys():
    meta = inputs.pop("meta")
    # budget = jnp.where(budget == 1, budget, budget // 2)
    inputs_meta = dict(
          budget=meta["k_eff"][0], # [MC]
          sources=meta["sources"],
          out_offsets=meta["out_offsets"][0],
          # class_valid_mask=jnp.transpose(meta["class_valid_mask"]), # [MC, N]
          # out_offsets=meta["out_offsets"][0],  # [MC]
      )
    # jax.debug.print("input kets after {} =={}", inputs_meta["budget"].shape, meta["k_eff"].shape)
  if task_cfg["subsel"]["val_anchors"]:
    if "meta" in val_batch.keys():
      meta = val_batch.pop("meta")
      val_meta = dict(sources=meta["sources"],)
    else:
      val_meta = dict(sources=jnp.ones(bs, dtype=jnp.int32),)
  else:
     val_meta = inputs_meta
  
  mt_train_batch, mt_val_batch = None, None
  aux = {}

  # grads, hidden = per_ex_grads(model, inputs, grad_layer, chunk_size=16)
  # hidden, delta_train = hidden
  '''
  delta -> [B, D]
  jacobian -> [B, D, #params]
  grad -> delta x jac -> [B, #params]
  (1, D), (D, #params) -> (1, #params)
  (delta1 x jac1) x (delta2 x jac2).T
  (delta1 x jac1) x (jac2.T x delta2.T)
  delta1 x (jac1 x jac2.T) x delta2.T
  (1,D) x ((D, #) x (# x D)) x (D, 1) 
  '''
  # train_vars = (grads, hidden, inputs, delta_train)
  # grads, mt_train_batch, loss_dict = process_grads(train_vars, moments["train"], 
  #                                   (model,project), (optimizer_last, optimizer_head), task_cfg["lowpass_train"],step, task_cfg, rng=rng)
  # # aux["gradapprox_loss_train"] = loss
  # updatedict("train", loss_dict, aux)

  # anchors = grads
  # if task_cfg["val_anchors"]:
  #   val_grads, hidden =  per_ex_grads(model, val_batch, grad_layer, chunk_size=8)
  #   hidden, delta_val = hidden
  #   val_vars = (val_grads, hidden, val_batch, delta_val)
  #   val_grads, mt_val_batch, loss_dict = process_grads(val_vars, moments["val"], 
  #                                                 (model,project), (optimizer_last, optimizer_head), task_cfg["lowpass_val"], 
  #                                                 step, task_cfg, rng=rng)
  #   # aux["gradapprox_loss_train"] = loss
  #   updatedict("val", loss_dict, aux)


  #   anchors = val_grads
  # with utils.time_measure("Gradient processing"):
  if mode not in ["full", "random"]:
    grads, anchors, aux, aux_mom = process_gradsv2(model, project, optimizer_head, cache_train, 
                    cache_val, inputs, val_batch, moments["train"], moments["val"], step, task_cfg, rng)
    mt_train_batch, mt_val_batch = aux_mom["moments_train"], aux_mom["moments_val"]
  
  if mode not in ["full", "random"] and task_cfg["subsel"]["domainwise"]:
    S_tt = gram_linear(grads)
    S_tv = gram_linear(grads, anchors)
    D_tt = 1- S_tt
    k_eff = inputs_meta["budget"]
    sources = inputs_meta["sources"]
    out_offsets = inputs_meta["out_offsets"]
    flag = False
    triggers = ["facloc", "gradnorm"]
    if mode in triggers: flag = True
    idx, _ = select_per_class(S_tt, S_tv, D_tt, ratio, sources, 
                     k_eff, out_offsets, optim_name=mode,
                     max_classes= num_train_sources, apply_source_mask_on_target=flag)
    jax.debug.print("conflicting-pairs={}", conflicting(grads, idx, ratio))
    jax.debug.print("domainwise {} {} \n tgt_mask={}", mode, idx, flag)
  
  elif mode == "full":
    idx = jnp.arange(bs)
    ratio = bs

  elif mode == "random":
    rng, key = jax.random.split(rng)
    idx = jax.random.permutation(key, jnp.arange(bs))[:ratio]

  elif mode == "gradnorm":
    norms = tree_sq_norm(grads)
    _, idx = jax.lax.top_k(norms, ratio)
    jax.debug.print("gradnorm {} {}", norms, jnp.sort(idx), )

  elif(mode == "facloc"):
    sims = gram_linear(grads, anchors).astype(jnp.float32)
    idx = facility_location_old2(sims, ratio)

  elif(mode == "greats"):
    # jax.debug.print("lr {}", lr)
    
    sims = gram_linear(grads, anchors).astype(jnp.float32)
    scores = jnp.mean(sims, axis=1)
    interaction_matrix = gram_linear(grads).astype(jnp.float32)
    lr1 = lr
    lr2 = lr**2
    idx = greats_selection(lr1*scores, lr2*interaction_matrix, ratio, limit=None)
    jax.debug.print("greats idx {}", idx)
    jax.debug.print("conflicting-pairs={}", conflicting(grads, idx, ratio))


  elif(mode == "joint"):
    sources = inputs_meta["sources"]
    src_v = val_meta["sources"]

    anchors = wt_mean(task_cfg, num_train_sources, src_v, anchors)
    
    budget = inputs_meta["budget"]
    # make new budget where classes with only 1 el are always selected
    # budget = jnp.where(hist == 1, jnp.zeros_like(budget), budget) # [MC]

    sims = gram_linear(grads, anchors).astype(jnp.float32)
    scores = jnp.mean(sims, axis=1)
    interaction_matrix = gram_linear(grads).astype(jnp.float32)
    interaction_matrix = interaction_matrix*(1 - jnp.eye(sims.shape[0], sims.shape[0]))
    lr = jax.lax.cond(lr > 1e-12, lambda _: lr, lambda _: 1e-12, None)
    lr1 = lr
    lr2 = lr**2
    # apgd_lr = 1e-4
    beta = task_cfg["curricullum"]["beta"]
    lamb = task_cfg["curricullum"]["lamb"]
    max_iters = task_cfg["subsel"]["apdg_iters"]
    budget = curr_budget(task_cfg, lamb, sources, budget, cls_meta["gain"], ratio, rng)
    apgd_lr = 1/jax.numpy.linalg.matrix_norm(interaction_matrix)
    mask, utilities, weights = joint_subsel(rng, sources, budget, lr1*scores, 
                                  apgd_lr, lr2*interaction_matrix ,cls_meta["prev_utils"], max_iters)
    
    final_utils = per_class_utilities(grads, anchors, src_v, num_train_sources, lr1, lr2, interaction_matrix, weights)
    # 1. Compute temp (same as before)
    temp = jnp.where(
        cls_meta["prev_utils"] == 0,
        0.0,
        (final_utils - cls_meta["prev_utils"]) / jnp.abs(cls_meta["prev_utils"])
    )

    # 2. Update gain only at src_v indices
    gain = cls_meta["gain"]

    # src_v must be an int array of class indices in [0, num_train_sources)
    # gain = gain.at[src_v].set(
    #     beta * temp[src_v] + (1.0 - beta) * gain[src_v]
    # )
    gain = beta * temp + (1.0 - beta) * gain

    cls_meta["gain"] = gain
    cls_meta["prev_utils"] = final_utils
    # cls_meta["prev_utils"] = utilities
    import sys
    jnp. set_printoptions(threshold=sys.maxsize)

    # jax.lax.cond(
    #   step % 8 == 0,
    #   lambda _: jax.debug.print("utilities #START\n{}\n#END", utilities[:ratio]),
    #   lambda _: None,
    #   None
    # )
    # jax.debug.print("weights {} {} {}",  weights, utilities.shape, ratio)

    idx = jnp.nonzero(mask, size=ratio)[0]

    jax.debug.print("Learning rate {}", apgd_lr)
    jax.debug.print("conflicting-pairs={}", conflicting(grads, idx, ratio))
    jax.debug.print("class_utils={}", final_utils)
    jax.debug.print("gain={}", cls_meta["gain"])
    jax.debug.print("src_v={}", src_v)
    jax.debug.print("prev_utils[0]: {}", cls_meta["prev_utils"][0])
    jax.debug.print("final_utils[0]: {}", final_utils[0])
    jax.debug.print("temp[0]: {}", temp[0])
    jax.debug.print("gain[0] before: {}", cls_meta["gain"][0])
    jax.debug.print("lowpass={}", task_cfg["grads"]["lowpass_adam"])
    # add els to mask where class size == 1
    # mask = jnp.where()
    cnt = jnp.bincount(sources, length=num_train_sources)
    jax.debug.print("joint idx \n bin={} \n bud={} {} \n idx={} #############################", 
                        cnt, budget, jnp.sum(budget), idx)

  
  rng, key = jax.random.split(rng)
  idx = jax.random.permutation(key, idx)
  subset = jax.tree.map(lambda x: x[idx], inputs)
  # subset = jax.tree.map(lambda x : x[idx], inputs)
  # loss_mask = jnp.zeros_like(norms).at[idx].set(True)
  # loss_mask = jnp.ones((ratio))
  loss_mask = jnp.zeros((bs)).at[idx].set(1)
  # subset = jax.tree.map(lambda x: x.block_until_ready(), subset)
  # state = nnx.state((model, project, optimizer_head))
  def choose_mt(mt_batch):
    mu, nu = None, None
    if mt_batch is not None:
      mu_batch, nu_batch = mt_batch
      mu = jax.tree.map(lambda x: x[idx].mean(0), mu_batch)
      nu = jax.tree.map(lambda x: x[idx].mean(0), nu_batch)
    return (mu, nu)
  
  # lp_val = task_cfg["grads"]["lowpass_val"]
  if task_cfg["grads"]["lowpass_adam"]:
    val_smooth = task_cfg["grads"]["val_smooth"]
    if val_smooth == "val": pass
    elif val_smooth == "train": mt_val_batch = mt_train_batch
  moments = {
    "train": choose_mt(mt_train_batch),
    "val": choose_mt(mt_val_batch),
  }
  # ret1, ret2 = _train_step(model, optimizer, inputs, _steps, loss_mask)
  return subset, moments, loss_mask, aux, cls_meta

def chunk_fn(model, inputs, task_cfg, rng,chunk_size=16):
  def stack(trees, axis=0): return jax.tree.map(lambda *xs: jnp.concat(xs, axis=axis), *trees)
  size = jax.tree.leaves(inputs)[0].shape[0]
  chunks = max(size//chunk_size, 1)
  out1, out2, out3 = [], [], []
  for i in range(chunks):
    start = i*chunk_size
    end = start + chunk_size
    sub = {}
    for k in inputs.keys():
      if k == "meta": continue
      sub[k] = inputs[k][start: end]
    ret1, ret2, ret3 = _chunk_per_ex_grads(model, task_cfg, rng, sub)
    out1.append(ret1)
    out2.append(ret2)
    out3.append(ret3)

  return stack(out1), stack(out2), stack(out3)

def per_ex_grads(
    model, step, inputs: Any, task_cfg, rng, chunk_size=16, moments=None, lowpass=False,
) -> ArrayLike | Tuple[ArrayLike, Any]: 
  # if "meta" in inputs.keys():
  #   inputs.pop("meta")
  
  grads, hidden, delta = chunk_fn(model, inputs, task_cfg, rng,chunk_size=chunk_size)
  dimred = task_cfg["grads"]["dimred"]
  meta = {
    "hidden": hidden,
    "delta": delta,
    "moments": None
  }

  if lowpass:
    # grads = pack_pytree(grads)

    grads, mu_hat, nu_hat = update_moments(grads, moments[0], moments[1], step)
    meta["moments"] = (mu_hat, nu_hat)
    grads = pack_pytree(grads)


  if dimred :
    # FIXME: assume lowpass and dimred cant be both true at same time for now
    assert not (dimred and lowpass)
    grads = pack_pytree(grads)
    grads = dimred_fft(rng, grads, model.config.embed_dim)

  return grads, meta

# @partial(nnx.jit, static_argnames=("task_cfg"))
def _chunk_per_ex_grads(
    model, task_cfg, rng, inputs: Any
) -> ArrayLike | Tuple[ArrayLike, Any]:
  input_tokens = inputs["input_tokens"]
  input_mask = inputs["input_mask"]
  positions = inputs["positions"]
  attention_mask = inputs["attention_mask"]

  grad_layer = task_cfg["grads"]["grad_layer"]
  grad_fn = nnx.value_and_grad(
    single_ex_loss,
    argnums=nnx.DiffState(0, partial(grad_filter, grad_layer)),
    has_aux=True
  )
  grad_fn = nnx.vmap(grad_fn, in_axes=(None, 0, 0, 0, 0))

  loss, grads = grad_fn(model, input_tokens,input_mask,positions,attention_mask)
  loss, (hidden, delta) = loss

  grads = jax.tree.map(lambda x: x.reshape(x.shape[0], -1).astype(jnp.float32), grads)

  # if task_cfg["grads"]["dimred"] == True:
  #   grads = pack_pytree(grads)
  #   grads = dimred_fft(rng, grads, model.config.embed_dim)
    
  return grads, hidden, delta


def dimred(grads, k):
  topk_tree = jax.tree.map(lambda x: jax.lax.top_k(x.mean(0), k)[1], grads)
  grads = jax.tree.map(lambda idxs, x: x[:, idxs], topk_tree, grads)
  return grads

@partial(
  nnx.jit,
  static_argnames=("k", "deterministic")
)
def dimred_fft(rng, grads, k, deterministic=False):
  if deterministic:
    rng = jax.random.key(0)
  rng, key = jax.random.split(rng)
  n = grads.shape[-1]
  knew = k//2
  signs, idx = make_fft_sketch_params(key, n, knew)

  return fft_sketch_real(grads, signs, idx)


def make_fft_sketch_params(key, n, k):
  # Random Rademacher signs: ±1
  key1, key2 = jax.random.split(key)
  signs = jax.random.rademacher(key1, (n,), dtype=jnp.int8).astype(jnp.float32)

  # Choose k frequency bins uniformly (don’t bias to low freqs)
  F = n // 2 + 1
  idx = jax.random.choice(key2, F, shape=(k,), replace=False)
  idx = jnp.sort(idx)
  return signs, idx

def fft_sketch_real(grads, signs, idx):
    n = grads.shape[-1]
    x = grads * signs

    spec = jnp.fft.rfft(x, axis=-1, norm="ortho")   # [..., F] complex
    picked = jnp.take(spec, idx, axis=-1)           # [..., k] complex

    # one-sided energy correction
    F = n // 2 + 1
    is_dc = (idx == 0)
    is_nyq = (idx == (F - 1)) & ((n % 2) == 0)
    w = jnp.where(is_dc | is_nyq, 1.0, jnp.sqrt(2.0)).astype(x.dtype)
    picked = picked * w

    y = jnp.concatenate([picked.real, picked.imag], axis=-1)  # [..., 2k]
    return jnp.sqrt(n / y.shape[-1]).astype(x.dtype) * y


def fft_sketch(grads, signs, idx):
  """
  grads: [..., n] real
  signs: [n] ±1
  idx: [k] indices into rfft bins (0..n//2)
  returns: [..., k] real sketch
  """
  n = grads.shape[-1]
  x = grads * signs  # broadcast over leading dims

  # rFFT -> take selected bins
  spec = jnp.fft.rfft(x, axis=-1, norm="ortho")             # [..., F] complex
  picked = jnp.take(spec, idx, axis=-1)       # [..., k] complex

  # map complex -> real k dims (simple choice: real part)
  # scale to roughly preserve dot products
  return jnp.sqrt(n / idx.shape[0]) * picked.real

def fft_sketch_2k(grads, signs, idx):
    n = grads.shape[-1]
    x = grads * signs
    spec = jnp.fft.rfft(x, axis=-1, norm="ortho")
    picked = jnp.take(spec, idx, axis=-1)  # [..., k] complex
    feats = jnp.concatenate([picked.real, picked.imag], axis=-1)  # [..., 2k]
    return jnp.sqrt(n / idx.shape[0]) * feats





def dimred_global_packed(grads, k: int):
  # grads leaves: [(B, D1), (B, D2), ...] -> concat: (B, Dtot)
  X = pack_pytree(grads)          # (B, Dtot)
  scores = X.mean(0)                            # (Dtot,)
  k = int(min(k, X.shape[1]))                   # static for jit
  idx = jax.lax.top_k(scores, k)[1]             # (k,)
  return jnp.take(X, idx, axis=1)               # (B, k)

def dimred_topk(grads, k: int):
  # grads leaves: [(B, D1), (B, D2), ...] -> concat: (B, Dtot)
  X = pack_pytree(grads)          # (B, Dtot)
  return jax.lax.top_k(X, k)[0]


@partial(nnx.jit)
def single_ex_loss(
    model,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
  """Per-sequence masked cross-entropy. If batch_mean=False, returns [B] vector."""
  input_tokens = input_tokens[None, :]
  input_mask = input_mask[None, :]
  positions = positions[None, :]
  attention_mask = attention_mask[None, :]
  per_seq_mean, (hidden, delta) = per_ex_loss(model, input_tokens, input_mask, positions, attention_mask)

  return per_seq_mean.sum(), (hidden[0], delta[0])


@partial(nnx.jit)
def per_ex_loss(
    model,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
  logits, _, hidden = model(input_tokens, positions, None, attention_mask, return_hidden=True)  # logits [B,T,V], hidden [B,T,D]
  logits = logits.astype(jnp.float32)

  logits = logits[:, :-1, :]                 # [B,T-1,V]
  target_tokens = input_tokens[:, 1:]        # [B,T-1]
  target_mask  = input_mask[:, 1:]           # [B,T-1]

  log_probs = jax.nn.log_softmax(logits, axis=-1)                      # [B,T-1,V]
  one_hot   = jax.nn.one_hot(target_tokens, logits.shape[-1], dtype=jnp.float32)  # [B,T-1,V]

  nll_tok = -jnp.sum(log_probs * one_hot, axis=-1) * target_mask       # [B,T-1]
  per_seq_sum   = jnp.sum(nll_tok, axis=-1)                            # [B]
  per_seq_count = jnp.maximum(jnp.sum(target_mask, axis=-1), 1e-8)     # [B]
  per_seq_mean  = per_seq_sum / per_seq_count                          # [B]

  # ----- delta: per-seq mean gradient w.r.t. final hidden states, aggregated over time -----
  probs = jnp.exp(log_probs)                                           # [B,T-1,V]
  delta = (probs - one_hot)                                            # [B,T-1,V]
  delta = jnp.sum(delta * target_mask[:, :, None], axis=1)             # [B,V]
  delta = delta / per_seq_count[:, None]                               # [B,V]

  # Map vocab-space gradient to hidden-space via vocab head weights.
  # If your head computes logits = hidden @ W (W is [D,V]), this is correct:
  if hasattr(model.config, "use_tied_embedding") and model.config.use_tied_embedding:
    W = model.embedder.input_embedding.value.T
  elif hasattr(model.config, "weight_tying") and model.config.weight_tying:
    W = model.embedder.input_embedding.value.T
  else:
    W = model.lm_head.w.value # [D,V]
  # W = model.lm_head.w.value # [D,V]
    
  delta = jnp.einsum("BV,DV->BD", delta, W) # [B,D]
  # delta = hidden[:, 0, :]
  
  Bs, D = hidden.shape[0], hidden.shape[-1]
  chex.assert_shape(delta, (Bs, D))

  # # Make alignment explicit: delta corresponds to hidden[:, :-1, :]
  # hidden = hidden[:, :-1, :]                                           # [B,T-1,D]

  return per_seq_mean, (hidden.astype(jnp.bfloat16), delta.astype(jnp.bfloat16))


def stable_entropy(gamma: jnp.ndarray) -> float:
    mask = gamma > 0
    # return -np.sum(gamma[mask] * np.log(np.maximum(gamma[mask], 1e-12)))
    return -jnp.sum(gamma * jnp.log(jnp.maximum(gamma, 1e-12)))



@partial(jax.jit, static_argnames=("optim_name", "out_len"))
def get_optim(optim_name, S_tt, S_tv, D_tt, out_len, limit, src_mask=None, tgt_mask=None):
    """
    Unified contract (enforced here, not per method):
      - Returns exactly length-M int32 vector of GLOBAL indices.
      - Positions >= limit (if given) are forced to -1.
      - Indices must be in [0, N); out-of-bounds are set to -1.
      - If src_mask is provided, any index with src_mask[idx] == False is set to -1.
      - When there are no candidates or limit <= 0, returns all -1.
      - Padding uses -1.

    Notes:
      - This wrapper passes masks and limit through to the underlying method,
        but *also* sanitizes its outputs to the contract above.
      - Assumes underlying methods already return a length-M vector (padded).
        If they don't, make them do so (static shape needed for JIT).
    """
    N = S_tt.shape[0]
    cols = jnp.arange(out_len, dtype=jnp.int32)

    # Default masks: True everywhere if None
    if src_mask is None:
        src_mask = jnp.ones((N,), dtype=bool)
    if tgt_mask is None:
        tgt_mask = jnp.ones((N,), dtype=bool)

    # Candidates exist?
    has_cand = jnp.logical_and(src_mask, tgt_mask).any()

    # Normalize limit: if None -> M; clamp to [0, out_len]
    # limit = jnp.where(limit is None, out_len, limit)
    limit = jnp.clip(jnp.asarray(limit, dtype=jnp.int32), 0, out_len)

    def _call_inner():
        if optim_name == "greats":
            out = greats_selection(
                S_tv.mean(1), S_tt, out_len,
                source_mask=src_mask, limit=limit
            )
        elif optim_name == "facloc":
            out = facility_location(
                S_tt, out_len, src_mask=src_mask, tgt_mask=tgt_mask, limit=limit
            )
        else:
            raise
        return out

    def _all_neg1():
        return jnp.full((out_len,), -1, dtype=jnp.int32)

    run_flag = jnp.logical_and(has_cand, limit > 0)
    sel = jax.lax.cond(run_flag, _call_inner, _all_neg1)      # [M], dtype may vary

    # ---- Contract enforcement (centralized) ----
    sel = jnp.asarray(sel, dtype=jnp.int32)                   # dtype normalize

    # Enforce limit positionally: slots >= limit are padding
    sel = jnp.where(cols < limit, sel, -1)

    # In-bounds check
    inb = (sel >= 0) & (sel < N)

    # Respect src_mask: only keep selections with src_mask=True
    # Guard gather with in-bounds mask to avoid indexing with -1
    src_ok = jnp.where(inb, src_mask[sel], False)

    # Final sanitize: invalid or masked-out -> -1
    sel = jnp.where(inb & src_ok, sel, -1)

    return sel



@partial(jax.jit, static_argnames=("out_len", "optim_name", "max_classes", "apply_source_mask_on_target"))
def select_per_class(
    S_tt, S_tv, D_tt,
    out_len: int,          # global selection budget
    sources,               # [N] int32 class id in [0..MC-1]
    k_eff,                 # [MC] int32, sum(k_eff) == out_len (or <= out_len if you allow slack)
    out_offsets,           # [MC] int32, prefix sum of k_eff (defines disjoint slices)
    optim_name: str,       # static
    max_classes: int,      # MC (static)
    apply_source_mask_on_target: bool
):
    """
    Returns:
      orders_out: [out_len] int32

    High-level logic:
      1) For each class c, run get_optim restricted to that class => sel[c, :]
         where sel[c, :] is fixed-length [out_len] padded with -1.
      2) Pack into a single [out_len] array using prefix-sum ownership:
           class c owns positions [out_offsets[c], out_offsets[c] + k_eff[c])
         For each output position p:
           find owning class c
           j = p - out_offsets[c]
           output[p] = sel[c, j]
      This avoids dynamic slice writes and avoids scatter collisions entirely.
    """

    # ---------- 1) Per-class optimizer runs (fixed shapes) ----------

    class_ids = jnp.arange(max_classes, dtype=sources.dtype)          # [MC]
    class_valid_mask = (sources[None, :] == class_ids[:, None])       # [MC, N]

    has_cand  = jnp.any(class_valid_mask, axis=1)                     # [MC]
    run_flags = has_cand & (k_eff > 0)                                # [MC]

    def per_class(k_i, mask_i, run_i):
        # If runnable: run optimizer on this class only.
        def _do():
            return get_optim(
                optim_name, S_tt, S_tv, D_tt,
                out_len,
                src_mask=mask_i,
                tgt_mask=mask_i if apply_source_mask_on_target else None,
                limit=k_i,
            )  # [out_len], padded with -1
        # If not runnable: all -1.
        def _zero():
            return jnp.full((out_len,), -1, dtype=jnp.int32)
        return jax.lax.cond(run_i, _do, _zero)

    sel = jax.vmap(per_class, in_axes=(0, 0, 0))(k_eff, class_valid_mask, run_flags)  # [MC, out_len]
    # jax.debug.print("sel {}", sel)


    # ---------- 2) Prefix-sum packing as a gather (JIT-safe) ----------

    # Each class owns a half-open interval:
    #   [start[c], end[c]) where start = out_offsets, end = out_offsets + k_eff.
    start = out_offsets.astype(jnp.int32)                # [MC]
    end   = (out_offsets + k_eff).astype(jnp.int32)      # [MC]

    # For each output position p, determine which class owns it.
    p = jnp.arange(out_len, dtype=jnp.int32)             # [out_len]

    # owns[c, p] = True iff p is in class c's slice.
    # If out_offsets/k_eff are correct prefix sums, each column p has exactly one True.
    owns = (p[None, :] >= start[:, None]) & (p[None, :] < end[:, None])  # [MC, out_len]

    # If sum(k_eff) == out_len, every p should be owned. If you allow slack, some may be unowned.
    has_owner = jnp.any(owns, axis=0)                    # [out_len]

    # Because slices are disjoint, argmax picks the unique owning class when has_owner is True.
    owner = jnp.argmax(owns, axis=0).astype(jnp.int32)   # [out_len] in [0..MC-1] (arbitrary if unowned)

    # Local index within that class slice: j = p - start[owner].
    j = (p - start[owner]).astype(jnp.int32)             # [out_len]

    # Gather the selected element: sel[owner[p], j[p]].
    gathered = sel[owner, j]                             # [out_len]

    # If a position is unowned, force -1. Also keep only valid indices (>=0).
    orders_out = jnp.where(has_owner, gathered, -1).astype(jnp.int32)

    return orders_out, None



def _l2_normalize(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

