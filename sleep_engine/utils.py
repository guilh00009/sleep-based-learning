import logging
import sys
import torch
import gc
from transformers import TrainerCallback

# Global flag
LOGGING_CONFIGURED = False

def setup_logging():
    """
    Sets up logging. Safe to call multiple times.
    """
    global LOGGING_CONFIGURED
    if LOGGING_CONFIGURED:
        return

    log_file = "app.log"
    # Force flushing to ensure logs appear immediately in Colab
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        file_handler = logging.FileHandler(log_file, mode='a')
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, PermissionError) as e:
        print(f"!!! LOG ERROR: {e}", file=sys.stderr)
        
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)
    
    LOGGING_CONFIGURED = True
    logging.info("Logging configured.")

class LoggingCallback(TrainerCallback):
    """
    A custom callback that logs the loss to app.log at each logging step.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss")
            if loss is not None:
                logging.info(f"Training Step: {state.global_step}, Loss: {loss:.4f}")
                # Força o Python a liberar referências do dicionário de logs imediatamente
                del logs 
                gc.collect()

def release_gpu_memory(model=None, tokenizer=None):
    """
    Completely releases memory using aggressive cleanup strategies specific for Colab.
    """
    logging.info("--> Releasing GPU memory (Nuclear Mode)...")
    
    # 1. Delete explicit objects
    if model is not None:
        # Move para CPU antes de deletar (ajuda a desfragmentar às vezes)
        try:
            model.cpu() 
        except: 
            pass
        del model
        
    if tokenizer is not None:
        del tokenizer

    # 2. CLEAR GHOST REFERENCES (Crucial para Colab/Jupyter)
    # Jupyter guarda o output das células em variáveis como _, __, Out.
    # Se você retornou o modelo em alguma célula anterior, ele ainda está na memória!
    try:
        import sys
        # Limpa tracebacks de erros anteriores (que seguram referências de objetos)
        if hasattr(sys, 'last_traceback'):
            del sys.last_traceback
        if hasattr(sys, 'last_type'):
            del sys.last_type
        if hasattr(sys, 'last_value'):
            del sys.last_value
            
        # Tenta limpar histórico do IPython se estiver rodando num notebook
        from IPython import get_ipython
        ip = get_ipython()
        if ip:
            # Limpa cache de output
            ip.displayhook.prompt_count -= 1
            if 'Out' in ip.user_ns:
                ip.user_ns['Out'] = {}
            # Limpa variáveis temporárias comuns do shell interativo
            for hidden in ['_', '__', '___']:
                if hidden in ip.user_ns:
                    del ip.user_ns[hidden]
    except ImportError:
        pass # Não estamos no Colab/Jupyter, segue a vida
    except Exception as e:
        logging.warning(f"Could not clear IPython cache: {e}")

    # 3. Garbage Collection Agressivo
    # Rodar várias vezes ajuda a pegar referências circulares
    for _ in range(3):
        gc.collect()
    
    # 4. Limpeza CUDA
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect() # Limpeza extra de memória compartilhada
    except:
        pass
        
    logging.info("--> GPU memory released.")
