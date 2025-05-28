# hf dpo suggested format
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def train_dpo_model(data_path: str, save_path: str):
    model_path = "./llama-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # === Load base model ===
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # === Enhanced LoRA config ===
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # === Load frozen reference model ===
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    # === Load full dataset from CSV (12000 samples) ===
    full_ds = load_dataset("csv", data_files={"train": data_path})["train"]
    train_ds = full_ds.select(range(min(10000, len(full_ds))))

    # === DPO trainer config ===
    dpo_cfg = DPOConfig(
        output_dir=save_path,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
        learning_rate=2e-5,
        beta=1.0,
        save_strategy="no",   # ✅ 不保存 checkpoint
        save_total_limit=0    # ✅ 确保不保留历史 checkpoint
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer
    )

    trainer.train()

    model = model.merge_and_unload()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\n✅ 模型训练完成并保存至：{save_path}")

if __name__ == "__main__":
    train_dpo_model(
        data_path="./datasets/dpo/decision_feeling_dpo.csv",
        save_path="./dpo_outputs/model_f_3B"
    )

    train_dpo_model(
        data_path="./datasets/dpo/decision_thinking_dpo.csv",
        save_path="./dpo_outputs/model_t_3B"
    )
