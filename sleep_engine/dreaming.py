import logging
import json
import traceback

def generate_dreams(dream_model, dream_tokenizer, formatted_history_string, dream_count=3):
    """
    Generates dreams with highly robust JSON parsing to handle various model errors.
    """
    logging.info("\n" + "="*50)
    logging.info(f"🌌 GENERATING {dream_count} DREAMS (Robust Parsing v2) 🌌")
    
    chat_for_template = [{"role": "user", "content": formatted_history_string}]
    final_prompt_for_model = dream_tokenizer.apply_chat_template(
        chat_for_template, tokenize=False, add_generation_prompt=True
    )
    logging.info(f"--- Final Prompt Sent to Model ---\n{final_prompt_for_model}\n-------------------")
    inputs = dream_tokenizer(final_prompt_for_model, return_tensors="pt").to(dream_model.device)
    
    generated_dreams = []
    for i in range(dream_count):
        logging.info(f"--> Generating dream {i + 1}/{dream_count}...")
        try:
            outputs = dream_model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=1.3)
            response_text = dream_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            logging.info(f"--- Full RAW Response (Dream {i+1}) ---\n{response_text}\n-------------------")
            
            clean_response = response_text.strip().replace("```json", "").replace("```", "").strip()
            start_index = clean_response.find('{')
            end_index = clean_response.rfind('}')
            if start_index != -1 and end_index != -1:
                json_core = clean_response[start_index : end_index + 1]
                if not json_core.startswith('['):
                    json_core = f"[{json_core}]"
                
                dream_conversation_turns = json.loads(json_core)
                logging.info(f"--> Successfully parsed after cleaning and wrapping.")
            else:
                raise ValueError("Could not find valid JSON objects in the response.")

            if isinstance(dream_conversation_turns, list) and all('user' in d and 'assistant' in d for d in dream_conversation_turns):
                generated_dreams.extend(dream_conversation_turns)
                logging.info(f"--> Dream {i + 1} PARSED SUCCESSFULLY.")
            else:
                logging.warning(f"--> WARNING: Dream {i + 1} was not a list of {{'user':..., 'assistant':...}} dicts. Skipping.")
        
        except Exception as e:
            logging.error(f"--> An unexpected error occurred during dream generation {i+1}: {e}")
            logging.error(traceback.format_exc())

    return generated_dreams
