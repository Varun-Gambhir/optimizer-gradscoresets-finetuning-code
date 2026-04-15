import torch
from torch.optim import Optimizer

def _matrix_inv_sqrt(mat: torch.Tensor, eps: float) -> torch.Tensor:
    dim = mat.shape[0]
    eye = torch.eye(dim, dtype=mat.dtype, device=mat.device)
    mat = mat + eps * eye
    eigvals, eigvecs = torch.linalg.eigh(mat)
    inv_sqrt_vals = torch.pow(torch.clamp(eigvals, min=eps), -0.5)
    return (eigvecs * inv_sqrt_vals.unsqueeze(0)) @ eigvecs.T

class ASGO(Optimizer):
    """
    PyTorch port of the ASGO optax optimizer.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, beta2=0.8, eps=1e-10, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(ASGO, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    if grad.ndim == 2:
                        dim = min(p.shape[0], p.shape[1])
                        state['precond'] = torch.zeros((dim, dim), dtype=p.dtype, device=p.device)
                    else:
                        state['precond'] = torch.zeros((1, 1), dtype=p.dtype, device=p.device)

                if grad.ndim < 2:
                    raise ValueError("ASGO expects ndim >= 2 parameters; use AdamW for 1D params.")

                m = state['momentum']
                precond = state['precond']

                next_m = momentum * m + (1.0 - momentum) * grad
                rows, cols = grad.shape
                
                if rows < cols:
                    gram = grad @ grad.T
                else:
                    gram = grad.T @ grad

                precond.mul_(beta2).add_(gram, alpha=1.0 - beta2)
                inv_precond = _matrix_inv_sqrt(precond, eps)
                eye = torch.eye(precond.shape[0], dtype=precond.dtype, device=precond.device)
                
                bad = ~torch.all(torch.isfinite(inv_precond))
                if bad:
                    inv_precond = eye

                if rows < cols:
                    update = inv_precond @ next_m
                else:
                    update = next_m @ inv_precond

                norm = torch.linalg.norm(update, ord='fro')
                scale = (0.2 * torch.sqrt(torch.tensor(rows * cols, dtype=update.dtype))) / (norm + 1e-12)
                update = update * scale

                # Decoupled weight decay followed by gradient step
                p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.add_(update, alpha=-lr)
                
                state['momentum'] = next_m

        return loss

class DASGO(Optimizer):
    """
    PyTorch port of the DASGO optax optimizer.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(DASGO, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    if p.ndim >= 2:
                        state['precond'] = torch.zeros((p.shape[1],), dtype=p.dtype, device=p.device)
                    else:
                        state['precond'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

                if grad.ndim < 2:
                    raise ValueError("DASGO expects ndim >= 2 parameters; use AdamW for 1D params.")

                if grad.ndim != 2:
                    raise ValueError("DASGO reference rule is implemented for 2D parameters only.")

                m = state['momentum']
                precond = state['precond']

                next_m = momentum * m + (1.0 - momentum) * grad
                gram_diag = torch.sum(grad * grad, dim=0)
                
                precond.mul_(beta2).add_(gram_diag, alpha=1.0 - beta2)

                update = next_m * torch.pow(precond + eps, -0.5)
                norm = torch.linalg.norm(update, ord='fro')
                scale = (0.2 * torch.sqrt(torch.tensor(grad.shape[0] * grad.shape[1], dtype=update.dtype))) / (norm + 1e-12)
                update = update * scale

                # Decoupled weight decay followed by gradient step
                p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.add_(update, alpha=-lr)
                
                state['momentum'] = next_m

        return loss

# Note: Shampoo is omited due to complexity but can be added similarly following the Newton root matrix inversion.
