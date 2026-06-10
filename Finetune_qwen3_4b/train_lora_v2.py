""" %%writefile /kaggle/working/train_lora.py

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from pathlib import Path
import shutil

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DATA_PATH = "/kaggle/input/datasets/azaki09/pure-burmese-corpus/pure_burmese_personality.json"
model_name = "Qwen/Qwen3-4B-Instruct-2507"
output_dir = "/kaggle/working/qwen3-burmese-nia-lora"

local_rank = int(os.environ.get("LOCAL_RANK", 0))

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

system_prompt = "You are Nia, a warm, helpful Burmese assistant. Reply naturally in Burmese."

def to_text(row):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = Dataset.from_list([to_text(x) for x in raw])
dataset = dataset.train_test_split(test_size=0.08, seed=42)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": local_rank},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model = prepare_model_for_kbit_training(model)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

training_args = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",
    max_length=512,

    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,

    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=1,

    fp16=False,
    bf16=False,
    max_grad_norm=0,

    optim="paged_adamw_8bit",

    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

trainer.train()

if trainer.is_world_process_zero():
    save_dir = Path(output_dir)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    shutil.make_archive(
        output_dir,
        "zip",
        save_dir
    )

    print("Saved LoRA adapter to:", save_dir)
    print("Zipped file:", output_dir + ".zip") """