import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login
from . import config

def load_chat_model(force_original=False):
    """
    Loads the chat model with intelligent adapter path detection.
    """
    logging.info("\n" + "="*50)
    logging.info("🔁 LOADING CHAT MODEL 🔁")
    
    # Login if needed
    if config.HF_TOKEN:
        login(token=config.HF_TOKEN)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer_local = AutoTokenizer.from_pretrained(config.MODEL_ID) 
    
    if tokenizer_local.pad_token is None:
        logging.info("--> Setting pad_token and resizing embeddings.")
        tokenizer_local.pad_token = tokenizer_local.eos_token
        base_model.resize_token_embeddings(len(tokenizer_local))
    
    final_model = base_model
    merged_adapter_path = os.path.join(config.CONTINUOUS_TRAINING_LORA, "merged_lora")
    
    if force_original:
        logging.info("--> `force_original=True`. Returning pristine base model.")
    elif os.path.exists(merged_adapter_path):
        logging.info(f"--> Found 'merged_lora' adapter at '{merged_adapter_path}'. Applying it.")
        final_model = PeftModel.from_pretrained(base_model, merged_adapter_path)
    elif os.path.exists(config.CONTINUOUS_TRAINING_LORA):
        logging.info(f"--> No 'merged_lora' found. Falling back to parent adapter at '{config.CONTINUOUS_TRAINING_LORA}'. Applying it.")
        final_model = PeftModel.from_pretrained(base_model, config.CONTINUOUS_TRAINING_LORA)
    else:
        logging.info("--> No continuous adapter found. Returning pristine base model.")
        
    logging.info("✅ Chat model is ready.")
    logging.info("="*50 + "\n")
    return final_model, tokenizer_local

def load_dream_model():
    """
    Loads the dream model, replicating the working example's loading method precisely.
    """
    logging.info("\n" + "="*50)
    logging.info("🧠 LOADING DREAM MODEL (Precise Replication) 🧠")
    
    if config.HF_TOKEN:
        login(token=config.HF_TOKEN)

    # Using device_map="cuda:0" to be exact.
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0", # Forcing exact device
    )
    dream_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    logging.info(f"--> Loading dream generation LoRA from: {config.DREAM_GEN_LORA_PATH}")
    dream_model = PeftModel.from_pretrained(base_model, config.DREAM_GEN_LORA_PATH)
    
    logging.info("✅ Dream generation model is ready.")
    return dream_model, dream_tokenizer
