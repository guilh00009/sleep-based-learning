import logging
import json
import random
import os
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from .utils import LoggingCallback

def train_on_dreams(model_to_train, tokenizer_local, dream_data):
    """
    Trains a NEW, fresh LoRA adapter and returns the exact path to the saved adapter checkpoint.
    """
    logging.info("\n" + "="*50)
    logging.info("🚀 TRAINING NEW 'SKILL' LoRA ADAPTER 🚀")

    random.shuffle(dream_data)
    logging.info("--> Shuffled the dream data to randomize training order.")
    logging.info(f"--> First 5 dreams AFTER shuffling:\n{json.dumps(dream_data[:5], indent=2)}")

    if tokenizer_local.pad_token is None:
        logging.info("Setting tokenizer.pad_token to tokenizer.eos_token")
        tokenizer_local.pad_token = tokenizer_local.eos_token

    formatted_turns = []
    for turn in dream_data:
        turn_text = (f"<start_of_turn>user\n{turn['user']}<end_of_turn>\n"
                     f"<start_of_turn>assistant\n{turn['assistant']}<end_of_turn>")
        formatted_turns.append(turn_text)
    train_dataset = Dataset.from_dict({"text": formatted_turns})
    logging.info(f"--> Created training dataset with {len(train_dataset)} SAMPLES.")

    peft_config = LoraConfig(
        r=256, lora_alpha=256, lora_dropout=0.15, bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    temp_adapter_path = "./temp-new-lora"
    training_args = SFTConfig(
        output_dir=temp_adapter_path,
        dataset_text_field="text",
        max_length=2048,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.03, optim="paged_adamw_8bit",
        bf16=True, tf32=True, logging_steps=1, save_strategy="epoch",
    )
    
    trainer = SFTTrainer(
        model=model_to_train, args=training_args, train_dataset=train_dataset,
        processing_class=tokenizer_local, peft_config=peft_config,
        callbacks=[LoggingCallback()]
    )
    
    logging.info("\n--> Starting QLoRA fine-tuning for the new adapter...")
    trainer.train()
    
    checkpoints = [d for d in os.listdir(temp_adapter_path) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise RuntimeError("SFTTrainer did not save any checkpoints. Training may have failed.")
    last_checkpoint_path = os.path.join(temp_adapter_path, max(checkpoints, key=lambda d: int(d.split('-')[-1])))
    logging.info(f"--> New skill adapter training complete. Final adapter is at: '{last_checkpoint_path}'")
    
    return last_checkpoint_path
