import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def train_dpo_model(data_path: str, save_path: str):
    model_path = "./llama-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    train_ds = load_dataset("json", data_files={"train": data_path})["train"].select(range(2000))

    dpo_cfg = DPOConfig(
        output_dir=save_path,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=8,
        learning_rate=2e-5,
        beta=0.5,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=os.path.join(save_path, "logs"),
        report_to="none"
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

    print(f"✅ 模型训练完成并保存至：{save_path}")

if __name__ == "__main__":
    train_dpo_model(
        data_path="./datasets/dpo/decision_feeling_dpo_flat.json",
        save_path="./dpo_outputs/model_f_3B"
    )

    train_dpo_model(
        data_path="./datasets/dpo/decision_thinking_dpo_flat.json",
        save_path="./dpo_outputs/model_t_3B"
    )
