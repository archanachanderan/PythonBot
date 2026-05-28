BASE_MODEL = "Qwen/Qwen1.5-0.5B-Chat"
ADAPTER_PATH = "./lora_adapter"
DATA_PATH = "../data/python_dataset.json"

MAX_NEW_TOKENS = 160   # ← halved
TEMPERATURE = 0.0         # ← greedy decoding
TOP_P = 1.0
DO_SAMPLE = False         # ← new flag

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 128      

SYSTEM_PROMPT = """
You are PythonBot, an expert Python tutor.
Give short but complete answers.
Limit explanations to 1-2 paragraphs.
Always finish the answer properly.
Use concise Python examples.
"""