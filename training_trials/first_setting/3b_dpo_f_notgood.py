import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

# 限制 CUDA 碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def main():
    model_path = "./llama-3B-Instruct"
    data_path = "./datasets/dpo/decision_feeling_dpo_flat.json"
    save_path = "./dpo_outputs/model_f_3B"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 加载 tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # === 4bit 量化配置 ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # 可选 bfloat16 / float16
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # === 主模型（GPU） + 4bit量化 + LoRA 注入 ===
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0}  # 自动放入第一个可用 GPU
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=4,  # 降低 r
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # 视模型结构定
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # === 参考模型放 CPU ===
    ref_model = AutoModelForCausalLM.from_pretrained(model_path).to("cpu")

    # === 加载数据集（扁平格式） ===
    train_ds = load_dataset("json", data_files={"train": data_path})["train"]

    # === DPO 配置 ===
    dpo_cfg = DPOConfig(
        output_dir="./dpo_outputs",
        per_device_train_batch_size=2,  # 显存友好
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        learning_rate=5e-5,
        beta=0.1,
        logging_steps=50,
        save_strategy="no"
    )

    # === 初始化 Trainer ===
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer
    )

    # === 开始训练 ===
    trainer.train()

    # === 合并 LoRA & 保存推理模型 ===
    model = model.merge_and_unload()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ 训练完成，模型已保存至：{save_path}")

if __name__ == "__main__":
    main()
