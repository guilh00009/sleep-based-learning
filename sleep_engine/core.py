import os
import shutil
import random
import json
import logging
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from . import config, utils, data, models, dreaming, training

# Global State
model = None
tokenizer = None
conversation_history = []

def sleep_and_dream():
    """
    Implements the continual learning strategy.
    """
    utils.setup_logging()
    global model, tokenizer, conversation_history
    
    try:
        is_first_run = not os.path.exists(config.CONTINUOUS_TRAINING_LORA)
        temp_adapter_parent_dir = "./temp-new-lora"

        if is_first_run:
            logging.info("--- Starting First Sleep Cycle ---")
        else:
            logging.info("--- Starting Subsequent Sleep Cycle (Averaging, Recalling, Grounding) ---")

        # Step 1: Release Memory
        logging.info("[Step 1/10] Releasing current chat model...")
        utils.release_gpu_memory(model, tokenizer)
        model = None
        tokenizer = None
        
        # Step 2: Generate New Dreams
        logging.info("[Step 2/10] Loading dream model for data generation...")
        dream_model, dream_tokenizer = models.load_dream_model()
        formatted_history = data.format_conversation_for_dreaming(conversation_history)
        new_dream_data = dreaming.generate_dreams(dream_model, dream_tokenizer, formatted_history, dream_count=17)
        del dream_model, dream_tokenizer
        utils.release_gpu_memory()

        if not new_dream_data:
            logging.warning("No new dreams generated. Reloading previous model and waking up.")
            model, tokenizer = models.load_chat_model()
            return "I had a dreamless sleep."

        # Step 3: Archive new dreams
        logging.info("[Step 3/10] Archiving new dreams to persistent storage...")
        data.append_dreams_to_storage(new_dream_data)

        # Step 4: Recall old dreams
        logging.info(f"[Step 4/10] Recalling {config.NUM_RECALLED_DREAMS} random dreams from the past...")
        recalled_dreams = data.sample_old_dreams(config.NUM_RECALLED_DREAMS)
        
        # Step 5: Sample grounding facts
        logging.info(f"[Step 5/10] Sampling {config.NUM_GROUNDING_FACTS} random grounding facts...")
        grounding_facts = data.sample_grounding_facts(config.GROUNDING_DATA_FILE, config.NUM_GROUNDING_FACTS)
        
        # Step 6: Combine all data sources
        logging.info("[Step 6/10] Combining new dreams, recalled dreams, and grounding facts...")
        combined_training_data = new_dream_data + recalled_dreams + grounding_facts
        random.shuffle(combined_training_data)
        logging.info(
            f"--> Created a combined training set with {len(new_dream_data)} new dreams, "
            f"{len(recalled_dreams)} recalled dreams, and {len(grounding_facts)} grounding facts. "
            f"Total: {len(combined_training_data)} samples."
        )

        logging.info(json.dumps(combined_training_data[:7],indent=2))
        
        # Step 7 & 8: Train New Adapter on Clean Base
        logging.info(f"[Step 7/10] Loading PRISTINE base model for clean training...")
        clean_base_model, clean_tokenizer = models.load_chat_model(force_original=True)
        logging.info("[Step 8/10] Training a new, pure 'skill' LoRA on the combined dataset...")
        new_adapter_path = training.train_on_dreams(clean_base_model, clean_tokenizer, combined_training_data)
        del clean_base_model, clean_tokenizer
        utils.release_gpu_memory()

        # Step 9: Load Base Model on CPU for Merging
        logging.info("[Step 9/10] Loading a fresh base model ON CPU for memory-safe merging...")
        merge_base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        merge_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        if merge_tokenizer.pad_token is None:
            merge_tokenizer.pad_token = merge_tokenizer.eos_token
            merge_base_model.resize_token_embeddings(len(merge_tokenizer))
        
        # Step 10: Perform Merge on CPU
        if is_first_run:
            logging.info("--> First run. Promoting new adapter to be the continuous adapter.")
            if os.path.exists(config.CONTINUOUS_TRAINING_LORA): shutil.rmtree(config.CONTINUOUS_TRAINING_LORA)
            shutil.move(new_adapter_path, config.CONTINUOUS_TRAINING_LORA)
            peft_model = PeftModel.from_pretrained(merge_base_model, config.CONTINUOUS_TRAINING_LORA)
        else:
            logging.info("--> Subsequent run. Performing 50/50 LINEAR weighted merge on CPU.")
            peft_model = PeftModel.from_pretrained(merge_base_model, config.CONTINUOUS_TRAINING_LORA, adapter_name="old_lora")
            peft_model.load_adapter(new_adapter_path, adapter_name="new_lora")
            peft_model.add_weighted_adapter(
                adapters=["old_lora", "new_lora"], weights=[0.5, 0.5],
                adapter_name="merged_lora", combination_type="linear"
            )

        logging.info("[Step 10b/10] Making adapter weights contiguous and saving...")
        adapter_to_save = "merged_lora" if not is_first_run else "default"
        peft_model.set_adapter(adapter_to_save)
        for param in peft_model.parameters():
            if param.requires_grad:
                param.data = param.data.contiguous()
        peft_model.save_pretrained(config.CONTINUOUS_TRAINING_LORA)
        logging.info(f"--> Saved the adapter '{adapter_to_save}' to '{config.CONTINUOUS_TRAINING_LORA}'.")

        # Final Step: Reload
        logging.info("[Final Step] Merge complete. Reloading final evolved model to GPU for chat...")
        del peft_model, merge_base_model
        model, tokenizer = models.load_chat_model()
        if os.path.exists(temp_adapter_parent_dir):
            shutil.rmtree(temp_adapter_parent_dir)

        logging.info("Waking up... New experiences have been integrated with past memories and core knowledge.")
        conversation_history.clear()
        return "I'm back. I've integrated our conversation with my memories and core knowledge. Let's start fresh."

    except Exception as e:
        logging.error("🔥🔥🔥 A CRITICAL ERROR OCCURRED DURING THE SLEEP/DREAM CYCLE 🔥🔥🔥")
        logging.error(traceback.format_exc())
        utils.release_gpu_memory(model, tokenizer)
        model, tokenizer = models.load_chat_model()
        conversation_history.clear()
        return f"A critical error occurred: {e}. I've reset to my last stable self."

def chat_with_gemma(message, history):
    utils.setup_logging()
    global model, tokenizer, conversation_history
    logging.info(f"Received message: '{message}'")
    if message.strip().lower() == "/sleep":
        if not conversation_history:
            logging.warning("Sleep command received but history is empty.")
            return "There is no conversation to dream about yet. Please talk to me first."
        logging.info("Sleep command received. Initiating dream sequence.")
        response_text = sleep_and_dream()
        return response_text
    
    messages_for_api = []
    for user_msg, assistant_msg in history:
        messages_for_api.append({"role": "user", "content": user_msg})
        messages_for_api.append({"role": "assistant", "content": assistant_msg})
    messages_for_api.append({"role": "user", "content": message})
    conversation_history = messages_for_api.copy()
    
    try:
        logging.info("Applying chat template and tokenizing...")
        inputs = tokenizer.apply_chat_template(
            messages_for_api, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        logging.info("Generating response from model...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95
            )
        input_length = inputs.shape[1]
        response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        logging.info(f"Model generated response: '{response_text[:100]}...'")
        conversation_history.append({"role": "assistant", "content": response_text})
    except Exception as e:
        logging.error("🔥🔥🔥 AN EXCEPTION OCCURRED DURING MODEL GENERATION 🔥🔥🔥")
        logging.error(traceback.format_exc())
        return f"Sorry, a critical error occurred. A full traceback has been written to app.log. Error: {e}"
    return response_text

def initialize_system():
    """
    Initializes the system by loading the chat model.
    """
    utils.setup_logging()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logging.info("CUDA_LAUNCH_BLOCKING enabled for precise error reporting.")
    
    global model, tokenizer
    if model is None:
        logging.info("No model loaded. Initializing chat model...")
        model, tokenizer = models.load_chat_model()
