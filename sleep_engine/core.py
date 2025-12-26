import os
import shutil
import random
import json
import logging
import traceback
import torch
import gc  # <--- IMPORTANTE: Garbage Collector
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from . import config, utils, data, models, dreaming, training

# Global State
model = None
tokenizer = None
conversation_history = []

def force_cleanup():
    """Limpa VRAM e RAM agressivamente."""
    gc.collect()
    torch.cuda.empty_cache()

# --- Adicione esta função helper no início ou junto com as outras funções ---
def clean_text(text):
    """
    Remove caracteres 'surrogates' que quebram o log/save do Python.
    Também força a codificação para UTF-8 ignorando erros.
    """
    if isinstance(text, str):
        # Primeiro encode/decode para remover bytes inválidos
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        # Depois encode/decode para remover surrogates isolados (o erro específico que vc teve)
        return text.encode('utf-16', 'surrogatepass').decode('utf-16').encode('utf-8', 'replace').decode('utf-8')
    return text

def clean_data_recursive(data):
    """Aplica a limpeza em listas e dicionários recursivamente."""
    if isinstance(data, dict):
        return {k: clean_data_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_recursive(item) for item in data]
    elif isinstance(data, str):
        return clean_text(data)
    else:
        return data

# --- Sua função sleep_and_dream atualizada ---
def sleep_and_dream():
    """
    Implements the continual learning strategy with VRAM safety for Colab.
    """
    utils.setup_logging()
    global model, tokenizer, conversation_history
    
    try:
        is_first_run = not os.path.exists(config.CONTINUOUS_TRAINING_LORA)
        temp_adapter_parent_dir = "./temp-new-lora"

        if is_first_run:
            logging.info("--- Starting First Sleep Cycle ---")
        else:
            logging.info("--- Starting Subsequent Sleep Cycle ---")

        # Step 1: Release Memory (Chat Model)
        logging.info("[Step 1/10] Releasing current chat model...")
        # Deletar explicitamente as referências globais
        del model
        del tokenizer
        model = None
        tokenizer = None
        force_cleanup() # Limpeza forçada
        
        # Step 2: Generate New Dreams
        logging.info("[Step 2/10] Loading dream model...")
        dream_model, dream_tokenizer = models.load_dream_model()
        
        formatted_history = data.format_conversation_for_dreaming(conversation_history)
        
        # OTIMIZAÇÃO: Reduzi dream_count de 17 para 5. 
        # 17 sonhos x 2048 tokens = estouro de memória no treinamento depois.
        logging.info("--> Generating dreams (Reduced count for Colab stability)...")
        raw_dream_data = dreaming.generate_dreams(
            dream_model, 
            dream_tokenizer, 
            formatted_history, 
            dream_count=11
        )


        logging.info("--> Sanitizing generated dreams (removing bad unicode/emojis)...")
        new_dream_data = clean_data_recursive(raw_dream_data)
        
        # Limpeza Imediata do Modelo de Sonho
        del dream_model
        del dream_tokenizer
        force_cleanup() 

        if not new_dream_data:
            logging.warning("No new dreams generated. Reloading...")
            model, tokenizer = models.load_chat_model()
            return "I had a dreamless sleep."

        # Step 3: Archive new dreams
        logging.info("[Step 3/10] Archiving new dreams...")
        data.append_dreams_to_storage(new_dream_data)

        # Step 4 & 5 & 6: Prepare Data
        # Reduzi levemente a quantidade de fatos antigos para economizar contexto
        recalled_dreams = data.sample_old_dreams(max(1, config.NUM_RECALLED_DREAMS // 2))
        grounding_facts = data.sample_grounding_facts(config.GROUNDING_DATA_FILE, max(1, config.NUM_GROUNDING_FACTS // 2))
        
        combined_training_data = new_dream_data + recalled_dreams + grounding_facts
        random.shuffle(combined_training_data)
        
        logging.info(f"--> Training set size: {len(combined_training_data)} samples.")

        # Step 7: Load Base Model for Training
        logging.info(f"[Step 7/10] Loading PRISTINE base model...")
        clean_base_model, clean_tokenizer = models.load_chat_model(force_original=True)
        
        # Step 8: Train
        logging.info("[Step 8/10] Training LoRA...")
        try:
            # Aqui é onde o Colab costuma travar. Certifique-se que o batch_size no training.py é 1.
            new_adapter_path = training.train_on_dreams(clean_base_model, clean_tokenizer, combined_training_data)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error("OOM during training. Clearing cache and trying to rescue...")
                del clean_base_model
                del clean_tokenizer
                force_cleanup()
                raise e # Relança erro para abortar o ciclo de sono com segurança
            else:
                raise e

        # Limpeza Pós-Treino
        del clean_base_model
        del clean_tokenizer
        force_cleanup()

        # Step 9: CPU Merging (Seguro para RAM, lento mas não trava GPU)
        logging.info("[Step 9/10] Loading model on CPU for merging...")
        merge_base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu" # CPU OBRIGATÓRIO
        )
        merge_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        
        if merge_tokenizer.pad_token is None:
            merge_tokenizer.pad_token = merge_tokenizer.eos_token
            merge_base_model.resize_token_embeddings(len(merge_tokenizer))
        
        # Step 10: Perform Merge
        if is_first_run:
            logging.info("--> Promoting new adapter.")
            if os.path.exists(config.CONTINUOUS_TRAINING_LORA): shutil.rmtree(config.CONTINUOUS_TRAINING_LORA)
            shutil.move(new_adapter_path, config.CONTINUOUS_TRAINING_LORA)
            peft_model = PeftModel.from_pretrained(merge_base_model, config.CONTINUOUS_TRAINING_LORA)
        else:
            logging.info("--> 50/50 Weighted Merge on CPU.")
            peft_model = PeftModel.from_pretrained(merge_base_model, config.CONTINUOUS_TRAINING_LORA, adapter_name="old_lora")
            peft_model.load_adapter(new_adapter_path, adapter_name="new_lora")
            peft_model.add_weighted_adapter(
                adapters=["old_lora", "new_lora"], weights=[0.5, 0.5],
                adapter_name="merged_lora", combination_type="linear"
            )

        adapter_to_save = "merged_lora" if not is_first_run else "default"
        peft_model.set_adapter(adapter_to_save)
        
        # Salvar
        peft_model.save_pretrained(config.CONTINUOUS_TRAINING_LORA)
        logging.info(f"--> Adapter saved.")

        # Limpeza Final antes de acordar
        del peft_model
        del merge_base_model
        if 'merge_tokenizer' in locals(): del merge_tokenizer
        force_cleanup()

        # Final Step: Reload
        logging.info("[Final Step] Waking up (Reloading Chat Model)...")
        model, tokenizer = models.load_chat_model()
        
        if os.path.exists(temp_adapter_parent_dir):
            shutil.rmtree(temp_adapter_parent_dir)

        conversation_history.clear()
        return "I'm back. Memories integrated."

    except Exception as e:
        logging.error(f"🔥🔥🔥 CRITICAL SLEEP ERROR: {e}")
        traceback.print_exc()
        
        # Tentativa de recuperação de desastre
        force_cleanup()
        
        # Se o modelo não estiver carregado, tenta carregar para não quebrar o chat
        if model is None:
            try:
                logging.info("Emergency reload of chat model...")
                model, tokenizer = models.load_chat_model()
            except:
                logging.error("Could not reload model during emergency recovery.")
        
        return f"I had a nightmare (Error: {e}). I've reset myself."

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
