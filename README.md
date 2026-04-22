# PythonBot — Domain-Specific Tutor

A fine-tuned Python programming tutor built with LLaMA-2-7B, LoRA, and a FastAPI + HTML frontend.

---

## Folder Structure

```
python-tutor-bot/
├── backend/
│   ├── app.py          # FastAPI server
│   ├── model.py        # Model loading + inference
│   ├── train.py        # LoRA fine-tuning
│   ├── dataset.py      # Dataset loading & formatting
│   ├── config.py       # Central config (model, LoRA, training)
│   └── requirements.txt
├── frontend/
│   └── index.html      # Chat UI (no framework needed)
├── data/
│   ├── python_dataset.json
│   └── python_dataset.csv
└── README.md
```

---

## Dataset Format

Use the **JSON** file (better for code snippets with quotes/commas).

```json
[
  {
    "instruction": "How do you define a function in Python?",
    "input": "",
    "output": "Use the `def` keyword:\n```python\ndef greet(name):\n    return f'Hello, {name}'\n```"
  },
  {
    "instruction": "Fix this code:",
    "input": "for i in range(10)\n    print(i)",
    "output": "Missing colon after range(10):\n```python\nfor i in range(10):\n    print(i)\n```"
  }
]
```

`input` can be empty `""` — the loader handles it.

---

## Setup

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Authenticate with Hugging Face (for LLaMA-2)

LLaMA-2 is a gated model. You need to:
1. Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Create a token at https://huggingface.co/settings/tokens
3. Login:

```bash
huggingface-cli login
# paste your token when prompted
```

### 3. (Optional) Fine-tune with LoRA

```bash
cd backend
python train.py
```

This saves the adapter to `./lora_adapter/`. Requires a GPU (≥16GB VRAM recommended, or use 4-bit quant on 8GB).

### 4. Start the backend

```bash
cd backend
python app.py
```

Server runs at http://localhost:8000

### 5. Open the frontend

Open `frontend/index.html` directly in your browser — or the server also serves it at http://localhost:8000

---

## Switching Models

Change `BASE_MODEL` in `config.py`:

```python
# LLaMA-2 7B (default)
BASE_MODEL = "meta-llama/Llama-2-7b-hf"

# Smaller / faster alternatives:
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
BASE_MODEL = "microsoft/phi-2"
BASE_MODEL = "google/gemma-7b-it"
```

---

## API Reference

### POST /chat

```json
{
  "message": "How do list comprehensions work?",
  "history": [],
  "use_lora": true
}
```

Response:
```json
{
  "response": "List comprehensions are...",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### GET /health

```json
{"status": "ok", "model": "PythonTutorBot"}
```

---

## LoRA Toggle (Frontend)

The UI has a **LoRA toggle** in the header:
- **ON** → uses fine-tuned LoRA adapter (requires `./lora_adapter/` to exist)
- **OFF** → uses base model with system prompt only (prompt engineering mode)

This lets you compare both approaches in the same UI.
