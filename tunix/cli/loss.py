"""Main entry point for PEFT training."""
import jax
import optax
from flax import nnx
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from functools import partial
from tunix.cli.beir import *
from tunix.sft import utils
from ott.geometry import pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn
import ott

# from tunix.cli import pot as myot

def default_loss_fn(
    model: nnx.Module,
    inputs,
) -> ArrayLike:
  """Default loss function for PEFT training."""
  input_tokens = inputs["input_tokens"]
  input_mask = inputs["input_mask"]
  positions = inputs["positions"]
  attention_mask = inputs["attention_mask"]
  
  logits, _ = model(input_tokens, positions, None, attention_mask)
  logits = logits.astype(jnp.float32)
  # Exclude the last step as it does not appear in the targets.
  logits = logits[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Convert the target labels to one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

  # Define the normalization factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the negative log likelihood (NLL) loss.
  # Equivalent to: optax.softmax_cross_entropy(logits, one_hot).mean()
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor, None


def prepare_input(inputs):
  pad_mask = inputs["input_mask"]
  tokens = inputs["input_tokens"]

  positions = utils.build_positions_from_mask(pad_mask)
  attention_mask = utils.make_causal_attn_mask(pad_mask)

  return {
    "input_tokens": tokens,
    "positions": positions,
    "attention_mask": attention_mask,
  }


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

def prepare_input_ret(inputs, pad_id, is_causal=True):
  tokens = inputs["input_tokens"]
  pad_mask = fix_pad_mask(tokens, pad_id)

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

@partial(nnx.jit)
def loss_unpacked(
    model,
    inputs,
    aux
) -> ArrayLike:
  """Per-sequence masked cross-entropy. If batch_mean=False, returns [B] vector."""
  # loss_mask = aux["loss_mask"]
  input_mask = inputs["input_mask"]
  # inputs = prepare_input(inputs)
  input_tokens = inputs["input_tokens"]
  positions = inputs["positions"]
  attention_mask = inputs["attention_mask"]

  logits, _, = model(input_tokens, positions, None, attention_mask)  # [B,T,V]
  logits = logits.astype(jnp.float32)
  # jax.debug.print("logits {}", logits.shape)
  logits = logits[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask  = input_mask[:, 1:]  # [B,T-1]

  log_probs = jax.nn.log_softmax(logits, axis=-1)                    # [B,T-1,V]
  one_hot   = jax.nn.one_hot(target_tokens, logits.shape[-1])        # [B,T-1,V]
  nll_tok   = -jnp.sum(log_probs * one_hot, axis=-1) * target_mask   # [B,T-1]

  per_seq_sum   = jnp.sum(nll_tok, axis=-1)                          # [B]
  per_seq_count = jnp.maximum(jnp.sum(target_mask, axis=-1), 1e-8)    # [B]
  per_seq_mean  = per_seq_sum / per_seq_count                        # [B]
  # per_seq_mean = per_seq_mean*loss_mask
  # loss = per_seq_mean.sum()/loss_mask.sum()
  aux = {}
  return per_seq_mean.mean(), aux
  # return loss, aux

def cosine_sim(qry, docs, eps=1e-12):
    q_norm = jnp.linalg.norm(qry, axis=-1, keepdims=True)
    d_norm = jnp.linalg.norm(docs, axis=-1, keepdims=True)

    # Mark valid vectors (non-zero finite norm)
    q_mask = (q_norm > eps) & jnp.isfinite(q_norm)
    d_mask = (d_norm > eps) & jnp.isfinite(d_norm)

    # Normalize only valid ones; leave others zeroed
    qry_normed = jnp.where(q_mask, qry / q_norm, 0.0)
    docs_normed = jnp.where(d_mask, docs / d_norm, 0.0)

    sims = jnp.einsum("qd,pd->qp", qry_normed, docs_normed)
    return sims


def gromov_wasserstein_ot(A, B):
  progress_fn = ott.utils.default_progress_fn()
  linear_solver = sinkhorn.Sinkhorn(max_iterations=1000, threshold=1e-9, progress_fn=None)
  solver = ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein(
    linear_solver, epsilon=0.1,
    # threshold=1e-9,
    threshold=1e-9, # before slow it was 3
    # max_iterations=128,
    max_iterations=50,
    store_inner_errors=True,
    # progress_fn=progress_fn,
  )
  solver = jax.jit(solver)
  
  geom_xx = ott.geometry.geometry.Geometry(A)
  geom_yy = ott.geometry.geometry.Geometry(B)
  prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy)


  out = solver(prob)
  coupling_matrix = out.matrix
  return  coupling_matrix, out.reg_gw_cost, {"n_iters": out.n_iters}

def get_gmvloss(scores_qq, scores_dd, scores_qd_pos, mode, sinkhorn_mode="sinkhorn"):
  temp2 = 0.02
  targets = jnp.arange(scores_qq.shape[0])
  meta = {}
  def t(a,b):
    b = jax.nn.softmax(b, axis=-1)
    return optax.safe_softmax_cross_entropy(a, b).mean()

  # default, BASELINE
  if mode == "none":
    return 0, None, meta

  # variance, BASELINE
  if mode == "mse":
    gmv_loss_hard = optax.safe_root_mean_squares(scores_qq - scores_dd, 0.0)**2
    return gmv_loss_hard, None, meta


  # grmov wasserstein, BASELINE
  if mode == "gmw":
    # gmv_plan = gromov_wasserstein(qry, pos)
    gmv_plan, gmv_cost, meta = gromov_wasserstein_ot(1-scores_qq, 1-scores_dd)
    # gmv_plan, gmv_cost = myot.entropic_gromov_wasserstein(1-scores_qq, 1-scores_dd, epsilon=0.05, method=sinkhorn_mode,
    #                                             max_iter=128, sinkhornIters=128)
    gmv_loss_soft = optax.softmax_cross_entropy_with_integer_labels(gmv_plan/temp2, targets).mean()
    return gmv_cost, gmv_plan, meta


  ########### bakchodi #############

  # jensen shannon divergence
  if mode == "jsd":
    gmv_loss = (t(scores_qq, scores_dd) + t(scores_dd, scores_qq))/2
    return gmv_loss, None

  

  # gromov monge gap
  if mode == "gmg":
    gmv_plan, gmv_cost, meta = gromov_wasserstein_ot(1-scores_qq, 1-scores_dd)
    gmv_loss_hard = optax.safe_root_mean_squares(scores_qq - scores_dd, 0.0)**2
    return jnp.abs(gmv_loss_hard - gmv_cost), gmv_plan, meta

  # fused grmov wasserstein
  if mode == "fgw_planreg":
    fgw_plan, fgw_cost = myot.entropic_fused_gromov_wasserstein(1-scores_qd_pos, 1-scores_qq, 1-scores_dd,
                                                        epsilon=0.05, method=sinkhorn_mode,
                                                max_iter=128, sinkhornIters=128)
    return fgw_cost, fgw_plan

def cka(K: jnp.ndarray, K_hat: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Centered Kernel Alignment between two Gram matrices."""
    n = K.shape[0]
    H = jnp.eye(n, dtype=K.dtype) - jnp.ones((n, n), dtype=K.dtype) / n

    Kc = H @ K @ H
    Lc = H @ K_hat @ H

    num = jnp.vdot(Kc, Lc)
    denom = jnp.linalg.norm(Kc) * jnp.linalg.norm(Lc) + eps
    return num / denom

def loss_fn_masked(
    model: nnx.Module,
    loss_mask, # B
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
  """Default loss function for PEFT training."""
  # logits, _, hidden = model(input_tokens, positions, None, attention_mask, return_hidden=True)
  logits, _ = model(input_tokens, positions, None, attention_mask)

  # Exclude the last step as it does not appear in the targets.
  logits = logits[:, :-1, :] # B, T, V
  target_tokens = input_tokens[:, 1:] # B, T
  target_mask = input_mask[:, 1:] # B, T
  target_mask = target_mask * loss_mask[..., None]

  # Convert the target labels to one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1]) # B, T, V
  
  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

  # Define the normalization factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the negative log likelihood (NLL) loss.
  # Equivalent to: optax.softmax_cross_entropy(logits, one_hot).mean()
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor


# @partial(nnx.jit, donate_argnames=("model", "inputs"))
def fwd_fn(model, inputs):
  return model(inputs["input_tokens"], inputs["positions"], None, inputs["attention_mask"], return_hidden=True)[2]


@partial(nnx.jit, donate_argnames=("model", "inputs"), static_argnames=("config"))
def fwd_fn_train(model, inputs, config):
  chunk_bs = config["train"]["remat_bs"]
  n_chunk = inputs["input_tokens"].shape[0]//chunk_bs
  if n_chunk == 1:
    return fwd_fn(model, inputs)
  
  chunked = jax.tree.map(lambda v: v.reshape((n_chunk, -1) +  v.shape[1:]), inputs)

  # TODO: cant bother tuning this
  policy = None
  # policy = jax.checkpoint_policies.dots_saveable

  def scan_f(_, sub_batch):
    return None, fwd_fn(model, sub_batch)
  _, embs_chunked = jax.lax.scan(nnx.remat(scan_f, policy=policy), None, chunked)

  return embs_chunked.reshape((-1,) + embs_chunked.shape[2:])

@partial(nnx.jit, static_argnames=("config"), donate_argnames=("model", "input"))
def get_embs(model, input, config):
  mode = config["emb_mode"]
  PAD_ID = config["pad_id"]
  def eos1(embs, attention_mask):
    # attention_mask: (B, T, T) or (B, 1, T, T) with causal + padding applied
    am = attention_mask
    while am.ndim > 3:  # squeeze broadcast dims if present
        am = jnp.squeeze(am, axis=1)
    # Rows beyond sequence length are all False; valid rows have some True.
    row_any = jnp.any(am, axis=-1)      # (B, T)
    lengths = jnp.sum(row_any, axis=1)  # (B,)
    last_idx = lengths - 1              # (B,)
    return embs[jnp.arange(embs.shape[0]), last_idx].astype(jnp.float32)      # (B, D)
  embs =  nnx.jit(fwd_fn, donate_argnames=("model", "inputs"))(model, input)
  # pad_mask: (B, T) True for real tokens, False for pad
  pad_mask = input["input_tokens"] != PAD_ID

  if mode == "eos":
    lengths = jnp.sum(pad_mask, axis=1)              # (B,)
    last_idx = lengths - 1                           # (B,)
    embs = embs[jnp.arange(embs.shape[0]), last_idx] # (B, D)

  elif mode == "mean":
    embs = embs*pad_mask[:,:, None] # (B,T,D) (B,T,1) -> (B,T,D)
    embs = embs.sum(1) # (B,D)
    lens = pad_mask.sum(1)
    lens = jnp.maximum(lens, 1)
    embs = embs/lens[:, None] 

  else:
    raise NotImplementedError
  embs = embs.astype(jnp.float32)
  # if mode == "nerf":
  #   eos = nerf_posenc(eos)
  return embs



def get_embs_train(model, input, config):
  mode = config["emb_mode"]
  PAD_ID = config["pad_id"]
  embs = fwd_fn_train(model, input, config)
  tokens = input["input_tokens"]
  tokens = tokens.reshape(-1, tokens.shape[-1])
  # pad_mask = gradcache.tree_unchunk(tokens) != PAD_ID
  # pad_mask: (B, T) True for real tokens, False for pad
  pad_mask = tokens != PAD_ID

  lengths = jnp.sum(pad_mask, axis=1)              # (B,)
  last_idx = lengths - 1                           # (B,)
  embs = embs[jnp.arange(embs.shape[0]), last_idx] # (B, D)

  embs = embs.astype(jnp.float32)
  return embs



@partial(nnx.jit, static_argnames=("config"), donate_argnames=("model", "inputs"))
def loss_retr(
    config,
    model,
    inputs,
    aux,
) -> ArrayLike:
  neg = None
  pad_id = config["pad_id"]
  inputs = {k: prepare_input_ret(inputs[k], pad_id, is_causal=True) for k in inputs}
  bs = inputs["query"]["input_tokens"].shape[0]
  # inputs = gradcache.tree_chunk(inputs, 128)
  qry, pos = inputs["query"], inputs["document"]
  if "negative" in inputs.keys():
    neg = inputs["negative"]
  temp = 0.02
  temp2 = 0.02
  alpha = 1
  # lamb = 1e-2
  step = aux["step"]

  lamb = config["loss"]["lambda"](step)
  
  qry = get_embs_train(model, qry, config)
  pos = get_embs_train(model, pos, config)
  if neg is not None:
    neg = get_embs_train(model, neg, config)
    docs = jnp.concat([pos, neg], axis=0)
  else:
    docs = pos

  scores_qd = cosine_sim(qry, docs)
  scores_qd_pos = cosine_sim(qry, pos)
  scores_qq = cosine_sim(qry, qry)
  scores_dd = cosine_sim(pos, pos)

  targets = jnp.arange(bs)

  ot_loss = optax.softmax_cross_entropy_with_integer_labels(scores_qd/temp, targets).mean()
  
  gmv_cost, plan, gmv_meta = get_gmvloss(scores_qq, scores_dd, scores_qd_pos, 
                               mode=config["loss"]["mode"], sinkhorn_mode="sinkhorn_log")
  # plan_loss = optax.softmax_cross_entropy_with_integer_labels(plan/temp2, jnp.arange(scores_qq.shape[0])).mean()
  gmv_loss = gmv_cost
  # gmv_loss, plan = 0, None
  loss = alpha*ot_loss
  if lamb is not None:
    loss += lamb*gmv_loss
  key = config["loss"]["mode"] + "_loss"
  # gold_disc = optax.softmax_cross_entropy_with_integer_labels(plan, jnp.arange(scores_qq.shape[0])).mean()
  # rand_disc = optax.softmax_cross_entropy_with_integer_labels(jnp.ones_like(plan), jnp.arange(scores_qq.shape[0])).mean()
  aux = {"ot_loss": ot_loss, key: gmv_loss, 
         "lamb": lamb if lamb is not None else 0, 
        #  "gold_disc": gold_disc,
        #  "rand_disc": rand_disc,
          **gmv_meta
  }
  # jax.debug.print("{}", aux)
  # if plan is not None:
  #   jax.debug.print("{} {} loss 2 {}",scores_qd.shape, aux, jnp.argmax(plan, axis=-1))
  return loss, aux