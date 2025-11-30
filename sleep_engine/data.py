import json
import os
import random
import logging
from . import config

def transform_history_for_dreaming(chat_history):
    """
    Transforms a chat history from [{"role": "user", "content": "..."}] format
    to [{"user": "...", "assistant": "..."}] turn-based format.
    """
    transformed_history = []
    # Iterate through the history, taking two items at a time (user and assistant)
    for i in range(0, len(chat_history) - 1, 2):
        user_turn = chat_history[i]
        assistant_turn = chat_history[i+1]
        
        # Ensure we have a matching pair
        if user_turn['role'] == 'user' and assistant_turn['role'] == 'assistant':
            transformed_history.append({
                "user": user_turn['content'],
                "assistant": assistant_turn['content']
            })
    return transformed_history

def format_conversation_for_dreaming(chat_history):
    """
    Transforms history and formats it as a multi-line string of PRETTY-PRINTED dicts.
    """
    logging.info("--> Original chat history format received.")
    transformed_history = transform_history_for_dreaming(chat_history)
    logging.info("--> Transformed history to 'user'/'assistant' turn-based format.")
    
    turn_strings = [json.dumps(turn, indent=2) for turn in transformed_history]
    return ",\n".join(turn_strings)

def append_dreams_to_storage(new_dreams):
    """
    Appends newly generated dreams to the persistent JSON storage file.
    """
    logging.info(f"--> Appending {len(new_dreams)} new dreams to '{config.DREAM_STORAGE_FILE}'...")
    try:
        if os.path.exists(config.DREAM_STORAGE_FILE) and os.path.getsize(config.DREAM_STORAGE_FILE) > 0:
            with open(config.DREAM_STORAGE_FILE, 'r', encoding='utf-8') as f:
                all_dreams = json.load(f)
        else:
            all_dreams = []
            
        all_dreams.extend(new_dreams)
        
        with open(config.DREAM_STORAGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_dreams, f, indent=2)
            
        logging.info(f"--> Dream storage now contains a total of {len(all_dreams)} dreams.")
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"--> FAILED to append dreams to storage. Error: {e}")

def sample_old_dreams(num_to_sample):
    """
    Loads all dreams from storage and returns a random sample.
    """
    if not os.path.exists(config.DREAM_STORAGE_FILE):
        logging.info("--> No dream storage file found. Cannot recall old dreams.")
        return []
        
    try:
        with open(config.DREAM_STORAGE_FILE, 'r', encoding='utf-8') as f:
            all_dreams = json.load(f)
        
        if not all_dreams:
            logging.info("--> Dream storage is empty. No dreams to recall.")
            return []

        if len(all_dreams) <= num_to_sample:
            logging.info(f"--> Recalling all {len(all_dreams)} dreams from storage.")
            return all_dreams
        
        recalled_dreams = random.sample(all_dreams, num_to_sample)
        logging.info(f"--> Recalled {len(recalled_dreams)} random dreams from the past.")
        return recalled_dreams
        
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"--> FAILED to sample old dreams. The storage file might be corrupted. Error: {e}")
        return []

def sample_grounding_facts(filepath, num_to_sample):
    """
    Loads a large grounding dataset and returns a random sample from it.
    """
    if not os.path.exists(filepath):
        logging.warning(f"--> Grounding data file not found at '{filepath}'. Skipping.")
        return []
        
    try:
        logging.info(f"--> Loading grounding data from '{filepath}'...")
        with open(filepath, 'r', encoding='utf-8') as f:
            all_facts = json.load(f)
        
        if not all_facts:
            logging.info("--> Grounding data file is empty. No facts to sample.")
            return []

        if len(all_facts) <= num_to_sample:
            logging.info(f"--> Sampling all {len(all_facts)} grounding facts available.")
            return all_facts
        
        grounding_facts = random.sample(all_facts, num_to_sample)
        logging.info(f"--> Sampled {len(grounding_facts)} random grounding facts.")
        return grounding_facts
        
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"--> FAILED to sample grounding facts. The file might be corrupted. Error: {e}")
        return []
