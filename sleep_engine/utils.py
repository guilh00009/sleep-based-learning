import logging
import sys
import torch
import gc
from transformers import TrainerCallback

# Global flag
LOGGING_CONFIGURED = False

def setup_logging():
    """
    Sets up logging. It's safe to call this multiple times; it will only run once per process.
    """
    global LOGGING_CONFIGURED
    if LOGGING_CONFIGURED:
        return

    log_file = "app.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        # Use 'a' (append) mode so we don't wipe the log on every setup
        file_handler = logging.FileHandler(log_file, mode='a')
        # Adding [%(process)d] to the format to see which process is logging
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(process)d] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, PermissionError) as e:
        print(f"!!! CRITICAL LOGGING ERROR: COULD NOT WRITE TO {log_file} !!! Error: {e}", file=sys.stderr)
        
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(process)d] %(message)s"))
    logger.addHandler(stream_handler)
    
    LOGGING_CONFIGURED = True
    logging.info("Robust logging configured for this process.")

class LoggingCallback(TrainerCallback):
    """
    A custom callback that logs the loss to our app.log file at each logging step.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        # This method is called by the Trainer every `logging_steps`.
        if logs is not None:
            # The 'logs' dictionary contains the metrics like loss, learning_rate, etc.
            loss = logs.get("loss")
            if loss is not None:
                # Use our globally configured logger to write the loss.
                logging.info(f"Training Step: {state.global_step}, Loss: {loss:.4f}")

def release_gpu_memory(model=None, tokenizer=None):
    """Completely releases the model and empties the GPU cache."""
    logging.info("--> Releasing GPU memory...")
    
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
        
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("--> GPU memory released.")
