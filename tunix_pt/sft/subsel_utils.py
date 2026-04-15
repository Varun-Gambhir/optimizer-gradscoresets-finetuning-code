import torch
import torch.nn.functional as F
from typing import Any, Tuple, NamedTuple, Optional

def _partition_accounting(selected_mask: torch.Tensor, part_id: torch.Tensor, budget: torch.Tensor):
    """
    used[p] = how many elements already selected in partition p
    rem[p]  = remaining capacity in partition p (budget - used)
    cnt[p]  = how many total elements exist in partition p
    """
    P = budget.shape[0]
    
    # bincount over part_id; selected_mask acts as 0/1 weights
    used = torch.bincount(
        part_id,
        weights=selected_mask.to(torch.float32),
        minlength=P,
    ).to(budget.dtype)

    rem = budget - used
    cnt = torch.bincount(part_id, minlength=P).to(budget.dtype)
    return used, rem, cnt

def _sample_uniform_from_mask(pick_mask: torch.Tensor):
    did_pick = torch.any(pick_mask)
    if not did_pick:
        return False, torch.tensor(0, dtype=torch.int32, device=pick_mask.device)
    
    indices = torch.nonzero(pick_mask).squeeze(1)
    # Pick a random index
    rand_idx = torch.randint(0, indices.shape[0], (1,), device=pick_mask.device)
    picked = indices[rand_idx[0]].to(torch.int32)
    return True, picked

def apgd_partitions(
    A: Tuple[torch.Tensor, torch.Tensor],
    part_id: torch.Tensor,
    max_iterations: int,
    learning_rate: float,
    selected_mask: torch.Tensor,
    random_init: bool = False
):
    """
    scores: [n]
    interaction_matrix: [n, n]
    """
    scores, interaction_matrix = A
    
    def project(w):
        w = torch.where(selected_mask, w, torch.tensor(0.0, device=w.device))
        return torch.clamp(w, min=0.0)
    
    w_init = torch.zeros_like(part_id, dtype=torch.float32)
    if random_init:
        w_init = torch.rand_like(w_init)
    
    w = project(w_init)
    utility = torch.full((max_iterations,), float('-inf'), dtype=torch.float32, device=w.device)
    y = w.clone()
    t = 1.0
    
    for iter_i in range(max_iterations):
        g = scores - (interaction_matrix @ y)
        
        # Ascent
        w_next = project(y + learning_rate * g)
        
        t_next = 0.5 * (1.0 + torch.sqrt(torch.tensor(1.0 + 4.0 * t * t)))
        y_next = w_next + ((t - 1.0) / t_next) * (w_next - w)
        
        _util = torch.sum(scores * w_next) - 0.5 * torch.sum(w_next * (interaction_matrix @ w_next))
        utility[iter_i] = _util
        
        w = w_next
        y = y_next
        t = t_next
        
    return w, utility

def select_one_partition_matroid_toprelu(selected_mask: torch.Tensor, part_id: torch.Tensor, budget: torch.Tensor, proxy: torch.Tensor):
    used, rem, cnt = _partition_accounting(selected_mask, part_id, budget)
    rem_i = rem[part_id]
    feasible = (~selected_mask) & (rem_i > 0)
    
    # shortlist sorted
    scores = torch.clamp(proxy, min=0.0)
    NEG = float('-1e30')
    key_score = torch.where(feasible, scores, torch.tensor(NEG, device=scores.device))
    
    idx_score = torch.argsort(key_score, descending=True)
    idx_part = torch.argsort(part_id[idx_score], descending=False) # Stable sort requires PyTorch > 1.13 logic, but argsort is usually stable enough if specified.
    sort_idx = idx_score[idx_part]
    
    part_s = part_id[sort_idx]
    feas_s = feasible[sort_idx]
    cap_s = rem[part_s].to(torch.int32)
    
    P = budget.shape[0]
    counts = torch.zeros((P,), dtype=torch.int32, device=part_id.device)
    short_s = torch.zeros_like(feas_s)
    
    for i in range(part_s.shape[0]):
        p = part_s[i]
        feas = feas_s[i]
        cap = cap_s[i]
        c = counts[p]
        take = feas & (c < cap)
        counts[p] += take.to(torch.int32)
        short_s[i] = take
        
    has_pos = torch.any(feasible & (proxy > 0.0))
    use_fallback = (~has_pos) & torch.any(feasible)
    # PyTorch doesn't have scores_relu static conditional here trivially, assuming False logic
    pick_s = short_s
    
    did_pick, picked_s = _sample_uniform_from_mask(pick_s)
    picked = sort_idx[picked_s] if did_pick else torch.tensor(0, dtype=torch.int32, device=proxy.device)
    
    new_selected_mask = selected_mask.clone()
    if did_pick:
        new_selected_mask[picked] = True
        
    picked_random = did_pick & use_fallback
    return new_selected_mask, did_pick, picked, picked_random

def joint_subsel(part_id: torch.Tensor, budget: torch.Tensor, scores: torch.Tensor, apgd_lr: float, interaction_matrix: torch.Tensor, max_iters: int):
    Bs = part_id.shape[0]
    weights = torch.zeros((Bs,), dtype=torch.float32, device=part_id.device)
    selected_mask = torch.zeros((Bs,), dtype=torch.bool, device=part_id.device)
    iters = int(torch.sum(budget).item())
    picked_idxs = torch.full((Bs,), -1, dtype=torch.int32, device=part_id.device)
    
    utilities = torch.full((Bs, max_iters), float('-inf'), dtype=torch.float32, device=part_id.device)
    n_random = 0
    n_opt = 0
    
    for i in range(iters):
        # Grad utility
        proxy = scores - (interaction_matrix @ weights)
        
        selected_mask, did_pick, picked, picked_random = select_one_partition_matroid_toprelu(
            selected_mask, part_id, budget, proxy
        )
        
        if did_pick:
            weights, utility = apgd_partitions((scores, interaction_matrix), part_id, max_iters, apgd_lr, selected_mask, random_init=True)
        else:
            utility = torch.full((max_iters,), float('-inf'), dtype=torch.float32, device=part_id.device)
            
        picked_idxs[i] = picked
        utilities[i] = utility
        
        if picked_random:
            n_random += 1
        elif did_pick:
            n_opt += 1
            
    return selected_mask, utilities, weights


def greats_selection(scores: torch.Tensor, interaction_matrix: torch.Tensor, out_len: int, source_mask: Optional[torch.Tensor]=None, limit: Optional[int]=None):
    """
    scores: (n,) 1D vector
    interaction_matrix: (n, n)
    out_len: int number of selections to make
    returns: (K,) int32 selected indices
    """
    if limit is None: limit = out_len
    W = interaction_matrix
    
    if source_mask is not None:
        scores = torch.where(source_mask, scores, torch.tensor(float('-inf'), device=scores.device))
        
    effective_k = min(out_len, out_len if limit is None else limit)
    
    selected = torch.full((out_len,), -1, dtype=torch.int32, device=scores.device)
    
    cur_scores = scores.clone()
    
    for i in range(effective_k):
        idx = torch.argmax(cur_scores)
        selected[i] = idx
        cur_scores = cur_scores - W[idx, :]
        cur_scores[idx] = float('-inf')  # Prevent reselection
        
    return selected

def facility_location(
    S: torch.Tensor,
    k: int,
    src_mask: Optional[torch.Tensor] = None,
    tgt_mask: Optional[torch.Tensor] = None,
    limit: Optional[int] = None
) -> torch.Tensor:
    """
    Facility-Location greedy with early stop and padding.
    Returns int32 array of length k.
    """
    n = S.shape[0]
    device = S.device
    
    src_mask = torch.ones((n,), dtype=torch.bool, device=device) if src_mask is None else src_mask
    tgt_mask = torch.ones((n,), dtype=torch.bool, device=device) if tgt_mask is None else tgt_mask
    
    max_possible = int(torch.sum(src_mask.to(torch.int32)).item())
    base_k = k if limit is None else min(k, limit)
    effective_k = min(base_k, max_possible)
    
    selected = torch.full((k,), -1, dtype=torch.int32, device=device)
    chosen = torch.zeros((n,), dtype=torch.bool, device=device)
    neg_inf = float('-inf')
    
    S_clean = S.clone()
    S_clean[torch.isnan(S_clean) | torch.isinf(S_clean)] = 0.0
    
    if effective_k == 0:
        return selected
    
    gains0 = (S_clean * tgt_mask.unsqueeze(1)).sum(dim=0)
    gains0 = torch.where(src_mask, gains0, torch.tensor(neg_inf, device=device))
    j0 = torch.argmax(gains0)
    
    selected[0] = j0
    chosen[j0] = True
    best = S[:, j0].clone()
    
    for t in range(1, effective_k):
        delta = S_clean - best.unsqueeze(1)
        delta = torch.where(tgt_mask.unsqueeze(1), delta, torch.tensor(neg_inf, device=device))
        delta = torch.where(src_mask.unsqueeze(0), delta, torch.tensor(neg_inf, device=device))
        
        gains = torch.clamp(delta, min=0.0).sum(dim=0)
        valid = src_mask & (~chosen)
        gains = torch.where(valid, gains, torch.tensor(neg_inf, device=device))
        
        j = torch.argmax(gains)
        
        best = torch.maximum(best, S[:, j])
        best = torch.where(tgt_mask, best, torch.tensor(neg_inf, device=device))
        chosen[j] = True
        selected[t] = j
        
    return selected

def tree_sq_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    sq = 0.0
    for x in tensors:
        sq = sq + torch.sum(x.to(torch.float32) ** 2, dim=-1)
    return sq

def pack_pytree(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 0:
        return torch.empty(0)
    flattened = [t.view(t.shape[0], -1) for t in tensors]
    return torch.cat(flattened, dim=1)

def gram_linear(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    if Y is None:
        Y = X
    if isinstance(X, list):
        gram = 0.0
        for x, y in zip(X, Y):
            x = x.to(torch.float32).view(x.shape[0], -1)
            y = y.to(torch.float32).view(y.shape[0], -1)
            gram = gram + (x @ y.T)
        return gram
    else:
        X = X.to(torch.float32).view(X.shape[0], -1)
        Y = Y.to(torch.float32).view(Y.shape[0], -1)
        return X @ Y.T

def conflicting(grads: torch.Tensor, idxs: torch.Tensor, ratio: int) -> int:
    idxs = idxs[:ratio]
    if isinstance(grads, list):
        grads = pack_pytree(grads)
    subset_grads = grads[idxs]
    kernel = gram_linear(subset_grads, subset_grads)
    n = kernel.shape[0]
    kernel = kernel * (1 - torch.eye(n, n, device=kernel.device))
    pairs = (torch.sum(kernel < 0)) / 2
    return int(pairs.item())
