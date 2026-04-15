import sys
import torch
from omegaconf import OmegaConf
import logging

from tunix_pt.cli.utils import model as model_utils
from tunix_pt.cli import optax_ext
from tunix_pt.sft import subset_trainer

# We use standard PyTorch Dataset/DataLoader to mock data for MVP
from torch.utils.data import DataLoader, Dataset

class DummyDictDataset(Dataset):
    def __init__(self, size=1000, seq_len=128, vocab_size=32000):
        self.size = size
        self.seq_len = seq_len
        self.vocab = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.randint(0, self.vocab, (self.seq_len,))
        }

class TokenizerWrapper:
    def __init__(self, tok):
        self._tokenizer = tok
    def pad_id(self):
        return self._tokenizer.pad_token_id
    def eos_id(self):
        return self._tokenizer.eos_token_id

def build_optimizer_pt(model, optimizer_config):
    opt_type = optimizer_config.get("opt_type", "adamw").lower()
    lr = optimizer_config.get("learning_rate", 2e-5)
    weight_decay = optimizer_config.get("weight_decay", 0.1)

    if opt_type == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "asgo":
        optim = optax_ext.ASGO(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "dasgo":
        optim = optax_ext.DASGO(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        logging.warning(f"Optimizer {opt_type} not directly supported in this MVP. Falling back to AdamW.")
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    return optim

def default_loss_fn(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    return outputs.loss

def main():
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python peft_main_pt.py <path_to_base_config.yaml>")
        return

    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    logging.info(f"Loaded config from {config_path}")
    
    # Nested task config
    if config.get("task_config", {}).get("config"):
        task_config_path = config["task_config"]["config"]
        try:
            task_cfg = OmegaConf.load(task_config_path)
            logging.info(f"Loaded task config from {task_config_path}")
        except FileNotFoundError:
            logging.warning(f"Task config {task_config_path} not found.")

    # 1. Model & Tokenizer
    model_config = config.get("model_config", {})
    tokenizer_config = config.get("tokenizer_config", {})
    
    model, tokenizer_path = model_utils.create_model(model_config, tokenizer_config)
    tokenizer = model_utils.create_tokenizer(tokenizer_config, tokenizer_path=tokenizer_path)

    # 2. Datasets
    from tunix.examples.data import ift_dataset as data_lib_ift
    
    # We follow the same extraction as the JAX pipeline.
    task_config = config.get("task_config", {}).get("config", {})
    if isinstance(task_config, str):
        try: 
            task_config = OmegaConf.load(task_config)
            config["task_config"]["config"] = task_config
        except Exception: 
            task_config = {}
            config["task_config"]["config"] = task_config
    
    dataset_name = config.get("dataset_name", "TIGER-Lab/MathInstruct")
    eval_split = config.get("eval_split", 0.005)
    cache_dir = config.get("cache_dir", None)
    max_target_length = config.get("max_target_length", 256)
    
    # The pipeline extracts specific configuration for subsets
    # We parse the subsel configuration manually if task_config exists
    if hasattr(task_config, "subsel") and hasattr(task_config.subsel, "enabled"):
        subsel = task_config.subsel.enabled
    else:
        subsel = config.get("subset_select", {}).get("enabled", False)
        
    buffer = config.get("subset_select", {}).get("buffer", 1) if subsel else 1
    
    _ratio = config.get("subset_select", {}).get("ratio", 1.0)
    subsel_bs = int(config.get("batch_size", 4) * buffer * _ratio)
    
    tokenizer_wrapped = TokenizerWrapper(tokenizer)
    
    train_ds, eval_ds, dev_ds, data_meta = data_lib_ift.create_datasets(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        global_batch_size=config.get("batch_size", 4) * buffer,
        eval_global_batch_size=config.get("eval_batch_size", 4),
        max_target_length=max_target_length,
        num_train_epochs=100,
        tokenizer=tokenizer_wrapped,
        split_ratio=eval_split,
        config=config,
        subsel_bs=subsel_bs
    )
    
    train_dl = train_ds
    eval_dl = eval_ds
    
    # 3. Optimizer
    optimizer_config = config.get("optimizer_config", {})
    optimizer = build_optimizer_pt(model, optimizer_config)

    # 4. Trainer
    training_config_dict = config.get("training_config", {})
    training_config = subset_trainer.TrainingConfigPT(
        eval_every_n_steps=training_config_dict.get("eval_every_n_steps", 10),
        max_steps=training_config_dict.get("max_steps", None),
        gradient_accumulation_steps=training_config_dict.get("gradient_accumulation_steps", 1)
    )

    trainer = subset_trainer.PeftTrainerPT(
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        train_loss_fn=default_loss_fn,
        eval_loss_fn=default_loss_fn,
        full_config=OmegaConf.to_container(config, resolve=True)
    )

    # 5. Run
    trainer.train(train_dl, num_train_sources=1, eval_dl=eval_dl)
    logging.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
