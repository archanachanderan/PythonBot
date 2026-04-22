# dataset.py — Loads your JSON dataset and formats it for training

import json
import pandas as pd
from datasets import Dataset
from config import DATA_PATH, SYSTEM_PROMPT


def load_json_dataset(path: str) -> Dataset:
    """
    Loads the JSON dataset. Expected format:
    [{"instruction": "...", "input": "...", "output": "..."}, ...]
    Input can be empty string "".
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted = []
    for item in data:
        instruction = item.get("instruction", "").strip()
        user_input  = item.get("input", "").strip()
        output      = item.get("output", "").strip()

        # Combine instruction + input (input may be empty)
        if user_input:
            user_turn = f"{instruction}\n\n{user_input}"
        else:
            user_turn = instruction

        # Format as a chat-style prompt
        prompt = (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{user_turn} [/INST] {output} </s>"
        )
        formatted.append({"text": prompt})

    return Dataset.from_list(formatted)


def load_csv_dataset(path: str) -> Dataset:
    """Alternative: load from CSV if you prefer."""
    df = pd.read_csv(path).fillna("")
    formatted = []
    for _, row in df.iterrows():
        instruction = str(row.get("instruction", "")).strip()
        user_input  = str(row.get("input", "")).strip()
        output      = str(row.get("output", "")).strip()
        user_turn   = f"{instruction}\n\n{user_input}" if user_input else instruction
        prompt = (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{user_turn} [/INST] {output} </s>"
        )
        formatted.append({"text": prompt})
    return Dataset.from_list(formatted)


# ✅ JSON is better than CSV here because:
#   - Handles multi-line code snippets without escaping issues
#   - No quoting conflicts with Python code containing commas/quotes
#   - Directly maps to Python dicts without type inference problems
if __name__ == "__main__":
    ds = load_json_dataset(DATA_PATH)
    print(f"Loaded {len(ds)} examples")
    print(ds[0]["text"][:300])