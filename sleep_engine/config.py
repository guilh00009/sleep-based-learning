import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "google/gemma-3-4b-it"
DREAM_GEN_LORA_PATH = "./dream-gen-lora-v4"
CONTINUOUS_TRAINING_LORA = "./continuous-training-lora"
DREAM_STORAGE_FILE = "old_dreams.json"
NUM_RECALLED_DREAMS = 15
GROUNDING_DATA_FILE = "grounding-dataset.json"
NUM_GROUNDING_FACTS = 30
