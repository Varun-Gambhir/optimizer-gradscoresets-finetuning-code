from typing import Any, Tuple
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

def create_tokenizer(tokenizer_config: dict[str, Any], tokenizer_path: str = None) -> PreTrainedTokenizer:
    """Loads a HuggingFace tokenizer based on the config."""
    if not tokenizer_path:
        tokenizer_path = tokenizer_config.get('tokenizer_path', None)
    if not tokenizer_path:
        raise ValueError("Tokenizer path must be specified.")
    
    auth_token = os.environ.get('HF_TOKEN')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=auth_token,
        trust_remote_code=True
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

def apply_lora_to_model(base_model: PreTrainedModel, lora_config: dict[str, Any]) -> PreTrainedModel:
    """Applies LoRA to the base HuggingFace model using peft."""
    logging.info('Applying LoRA using config %r', lora_config)
    
    # Map the regex module_path from JAX config to PyTorch target modules structure
    # e.g., '.*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj' -> q_proj, k_proj, v_proj...
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    r = lora_config.get('rank', 16)
    alpha = lora_config.get('alpha', 2.0)
    
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=target_modules
    )
    
    # We ignore NF4 weight_qtype parsing for the MVP to keep dynamic setup clean,
    # but normally one would wrap the model load with BitsAndBytesConfig
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model

def create_model(
    model_config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    mesh: Any = None,
) -> Tuple[PreTrainedModel, str]:
    """Creates a PyTorch model and determines the tokenizer path based on the model config.
    
    Args:
        model_config: Dictionary containing 'model_name', 'model_source', 'model_id'
        tokenizer_config: Dictionary containing 'tokenizer_path'
        mesh: Ignored in PyTorch implementation.
    
    Returns:
        model: PyTorch native model (potentially wrapped with PEFT LoRA)
        tokenizer_path: Extracted tokenizer path
    """
    model_id = model_config['model_id']
    auth_token = os.environ.get('HF_TOKEN')
    
    logging.info(f"Loading PyTorch AutoModelForCausalLM from: {model_id}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # If qtype is defined in lora, simulate BitsAndBytes (8-bit generic fallback)
    load_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "token": auth_token,
        "trust_remote_code": True,
    }
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **load_kwargs
    )
    
    tokenizer_path = tokenizer_config.get('tokenizer_path', model_id)
    
    if model_config.get('lora_enabled', False) and model_config.get('lora_config'):
        base_model = apply_lora_to_model(base_model, model_config['lora_config'])
    else:
        logging.info('Training with Full Weight')
        
    if model_config.get('model_display', False):
        print(base_model)
        
    return base_model, tokenizer_path
