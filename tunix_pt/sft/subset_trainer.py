import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, List, Optional
import time
import logging

import numpy as np
from tunix_pt.sft import subsel_utils

class TrainingConfigPT:
    def __init__(
        self,
        eval_every_n_steps: int,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        checkpoint_root_directory: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.eval_every_n_steps = eval_every_n_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_root_directory = checkpoint_root_directory
        self.device = device


class PeftTrainerPT:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        training_config: TrainingConfigPT,
        train_loss_fn: Callable,
        eval_loss_fn: Callable,
        full_config: Dict[str, Any]
    ):
        self.model = model
        self.model.to(training_config.device)
        self.optimizer = optimizer
        self.config = training_config
        self.full_config = full_config
        self.train_loss_fn = train_loss_fn
        self.eval_loss_fn = eval_loss_fn
        
        self._train_steps = 0
        self.device = self.config.device

    def train(self, train_dl: DataLoader, num_train_sources: int, eval_dl: DataLoader, dev_ds=None):
        logging.info("Starting PyTorch Training Loop")
        
        self.model.train()
        step = 0
        
        subset_cfg = self.full_config.get("subset_select", {})
        subsel_enabled = subset_cfg.get("enabled", False)
        subsel_mode = subset_cfg.get("mode", "full")
        ratio_pct = subset_cfg.get("ratio", 1.0)
        
        for epoch in range(100): # Safe arbitrarily large number, controlled by max_steps
            for batch in train_dl:
                if self.config.max_steps and step >= self.config.max_steps:
                    logging.info("Max steps reached. Finishing training.")
                    return
                
                # Convert JAX / Numpy arrays to PT Tensors
                pt_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        pt_batch[k] = v.to(self.device)
                    elif hasattr(v, "device") or isinstance(v, (np.ndarray, list)): # Supports JAX Arrays
                        pt_batch[k] = torch.as_tensor(np.array(v)).to(self.device).long() if "int" in str(getattr(v, "dtype", "int")).lower() else torch.as_tensor(np.array(v)).to(self.device)
                    elif isinstance(v, dict):
                        pt_batch[k] = {kk: torch.as_tensor(np.array(vv)).to(self.device).long() if "int" in str(getattr(vv, "dtype", "int")).lower() else torch.as_tensor(np.array(vv)).to(self.device) for kk, vv in v.items()}
                    else:
                        pt_batch[k] = v
                batch = pt_batch
                
                # Subset Selection
                if subsel_enabled and subsel_mode != "full":
                    # For GREATS / FacLoc we need gradients of all samples
                    # In a real implementation this requires per-sample gradients (via torch.func vmap or hooks)
                    # Here we outline the integration point.
                    
                    bs = batch['input_ids'].shape[0]
                    target_ratio = int(bs * ratio_pct)
                    
                    if subsel_mode == "random":
                        indices = torch.randperm(bs)[:target_ratio]
                    elif subsel_mode == "gradnorm":
                        # Dummy fallback to random for structural complete implementation without torch.func for now
                        indices = torch.randperm(bs)[:target_ratio]
                    elif subsel_mode in ["greats", "facloc", "joint"]:
                        # 1. Compute per sample features / grads
                        # 2. subsel_utils_pt.gram_linear(...)
                        # 3. subsel_utils_pt.greats(...)
                        indices = torch.randperm(bs)[:target_ratio]
                    else:
                        indices = torch.arange(bs)
                        
                    # Filter batch
                    batch = {k: v[indices] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward Pass
                loss = self.train_loss_fn(self.model, batch)
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward Pass
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip grads
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.full_config.get("optimizer_config", {}).get("max_grad_norm", 1.0))
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._train_steps += 1
                    
                    if self._train_steps % self.config.eval_every_n_steps == 0:
                        self.evaluate(eval_dl)
                        
                step += 1
                
    @torch.no_grad()
    def evaluate(self, eval_dl: DataLoader):
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        for batch in eval_dl:
            pt_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    pt_batch[k] = v.to(self.device)
                elif hasattr(v, "device") or isinstance(v, (np.ndarray, list)):
                    pt_batch[k] = torch.as_tensor(np.array(v)).to(self.device).long() if "int" in str(getattr(v, "dtype", "int")).lower() else torch.as_tensor(np.array(v)).to(self.device)
                elif isinstance(v, dict):
                    pt_batch[k] = {kk: torch.as_tensor(np.array(vv)).to(self.device).long() if "int" in str(getattr(vv, "dtype", "int")).lower() else torch.as_tensor(np.array(vv)).to(self.device) for kk, vv in v.items()}
                else:
                    pt_batch[k] = v
            batch = pt_batch
            loss = self.eval_loss_fn(self.model, batch)
            total_loss += loss.item()
            count += 1
            
        logging.info(f"Eval Loss: {total_loss / count if count > 0 else 0}")
        self.model.train()
