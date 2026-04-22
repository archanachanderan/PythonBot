# model.py — Model loading and inference (supports both base & fine-tuned)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os
from config import BASE_MODEL, ADAPTER_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, SYSTEM_PROMPT


class PythonTutorModel:
    def __init__(self, use_lora: bool = True):
        """
        use_lora=True  → loads base model + your fine-tuned LoRA adapter
        use_lora=False → loads base model with prompt engineering only
        """
        self.use_lora = use_lora and os.path.exists(ADAPTER_PATH)
        print(f"🔧 Loading model | LoRA adapter: {self.use_lora}")
        self._load()

        self.model.eval()
        print("Device:", self.model.device)

    def _load(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_PATH if self.use_lora else BASE_MODEL
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )

        if self.use_lora:
            self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)
            print("✅ LoRA adapter loaded")
        else:
            self.model = base
            print("✅ Base model loaded (prompt engineering mode)")

        self.model.eval()

    def build_prompt(self, user_message: str, history: list[dict]) -> str:
        """
        Builds a LLaMA-2 chat-style prompt from conversation history.
        history = [{"role": "user"|"assistant", "content": "..."}]
        """
        prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        for i, msg in enumerate(history):
            if msg["role"] == "user":
                if i == 0:
                    prompt += f"{msg['content']} [/INST] "
                else:
                    prompt += f"<s>[INST] {msg['content']} [/INST] "
            else:
                prompt += f"{msg['content']} </s>"
        prompt += f"<s>[INST] {user_message} [/INST] "
        return prompt

    @torch.inference_mode()
    def generate(self, user_message: str, history: list[dict] = []) -> str:
        history = history[-2:]
    # build messages list for Qwen
        messages = []

    # system prompt
        messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })

    # history
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # new user message
        messages.append({
        "role": "user",
        "content": user_message
    })

    # apply Qwen chat template
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

        inputs = self.tokenizer(
        text,
        return_tensors="pt"
    ).to(self.model.device)

        outputs = self.model.generate(
    **inputs,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    do_sample=True,
    repetition_penalty=1.1,
    use_cache=True,
    pad_token_id=self.tokenizer.eos_token_id,
)

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        response = self.tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()

        if response == "":
            response = "Model generated empty response."

        return response

# Singleton — loaded once when the server starts
_model_instance = None

def get_model(use_lora: bool = True) -> PythonTutorModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = PythonTutorModel(use_lora=use_lora)
    return _model_instance