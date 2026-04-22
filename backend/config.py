# config.py — Central configuration for the entire project

BASE_MODEL = "Qwen/Qwen1.5-0.5B-Chat"   # swap with any HF model
ADAPTER_PATH = "./lora_adapter"             # saved after fine-tuning
DATA_PATH = "../data/python_dataset.json"   # JSON chosen over CSV (see note below)
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.3
TOP_P = 0.9

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# Training
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 48

SYSTEM_PROMPT = """You are PythonBot, an expert Python programming tutor.
Give concise but complete answers.
Use short explanations followed by examples.
Avoid unnecessary long paragraphs.
Always format code in proper Python syntax blocks."""