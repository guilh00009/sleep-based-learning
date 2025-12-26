import logging
import json
import traceback
import torch # Importante para limpar a VRAM
import gc    # Importante para o Garbage Collector

def generate_dreams(dream_model, dream_tokenizer, formatted_history_string, dream_count=3):
    """
    Versão otimizada para evitar travamento do Colab (Gerenciamento de VRAM).
    """
    logging.info("\n" + "="*50)
    logging.info(f"🌌 GENERATING {dream_count} DREAMS (VRAM Safe Mode) 🌌")
    
    chat_for_template = [{"role": "user", "content": formatted_history_string}]
    
    # Preparar o prompt
    final_prompt_for_model = dream_tokenizer.apply_chat_template(
        chat_for_template, tokenize=False, add_generation_prompt=True
    )
    
    logging.info(f"--- Prompt Size check ---")
    inputs = dream_tokenizer(final_prompt_for_model, return_tensors="pt").to(dream_model.device)
    logging.info(f"Input tokens: {inputs['input_ids'].shape[1]}")

    generated_dreams = []
    
    for i in range(dream_count):
        logging.info(f"--> Generating dream {i + 1}/{dream_count}...")
        
        # Variável para segurar os outputs, inicializada como None
        outputs = None 
        
        try:
            # Geração
            with torch.no_grad(): # Economiza muita memória não calculando gradientes
                outputs = dream_model.generate(
                    **inputs, 
                    max_new_tokens=2048, 
                    do_sample=True, 
                    temperature=1.3,
                    pad_token_id=dream_tokenizer.eos_token_id # Boa prática evitar warnings
                )
            
            # Decodificação (pega apenas a parte nova)
            response_text = dream_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # --- LIMPEZA DE MEMÓRIA IMEDIATA (CRUCIAL) ---
            del outputs
            torch.cuda.empty_cache()
            gc.collect()
            # ---------------------------------------------

            logging.info(f"--- Response Generated (Dream {i+1}) ---")
            
            # Limpeza da String (Parse JSON)
            clean_response = response_text.strip().replace("```json", "").replace("```", "").strip()
            start_index = clean_response.find('{')
            end_index = clean_response.rfind('}')
            
            dream_conversation_turns = [] # Default caso falhe
            
            if start_index != -1 and end_index != -1:
                json_core = clean_response[start_index : end_index + 1]
                # Tenta corrigir se o modelo não colocou colchetes de lista
                if not json_core.startswith('['):
                    json_core = f"[{json_core}]"
                
                try:
                    dream_conversation_turns = json.loads(json_core)
                    logging.info(f"--> Successfully parsed JSON.")
                except json.JSONDecodeError:
                    logging.warning("--> JSON Parse Error (Bad formatting from model).")
            else:
                logging.warning("--> Could not find valid JSON brackets.")

            # Validação da estrutura
            if isinstance(dream_conversation_turns, list) and len(dream_conversation_turns) > 0:
                # Verifica se o primeiro item tem as chaves corretas
                if all(k in dream_conversation_turns[0] for k in ('user', 'assistant')):
                    generated_dreams.extend(dream_conversation_turns)
                    logging.info(f"--> Dream {i + 1} ADDED.")
                else:
                    logging.warning("--> Structure invalid (missing keys).")
            
        except Exception as e:
            logging.error(f"--> Error in dream {i+1}: {e}")
            logging.error(traceback.format_exc())
            
        finally:
            # Garante limpeza mesmo se der erro no try
            if 'outputs' in locals() and outputs is not None:
                del outputs
            torch.cuda.empty_cache()
            gc.collect()

    return generated_dreams
