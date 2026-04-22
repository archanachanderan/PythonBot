# train.py — LoRA fine-tuning with PEFT + Transformers (CPU friendly)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from dataset import load_json_dataset
from config import *


def tokenize(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )


def train():
    print("📦 Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # CPU-friendly model loading (NO quantization)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )

    # Important for CPU training
    model.config.use_cache = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("📂 Loading dataset...")
    dataset = load_json_dataset(DATA_PATH)
    tokenized = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=ADAPTER_PATH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=10,
        max_steps=1000,
        learning_rate=LEARNING_RATE,
        fp16=False,   # MUST be False for CPU
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("🚀 Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("⏹️ Training interrupted — saving model...")

    trainer.save_model(ADAPTER_PATH)
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)

    print(f"✅ LoRA adapter saved to {ADAPTER_PATH}")


if __name__ == "__main__":
    train()