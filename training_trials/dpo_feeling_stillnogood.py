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
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # === 主模型（GPU） + 4bit量化 + LoRA 注入 ===
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,  # 更高表达能力
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 覆盖更完整
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # === 参考模型放 CPU（默认 fp32）===
    ref_model = AutoModelForCausalLM.from_pretrained(model_path).to("cpu")

    # === 加载数据集 ===
    train_ds = load_dataset("json", data_files={"train": data_path})["train"]

    # === DPO 配置 ===
    dpo_cfg = DPOConfig(
        output_dir=save_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=6,                # 提高训练轮数
        learning_rate=5e-5,
        beta=0.05,                         # 更强偏好压制
        logging_steps=25,
        save_strategy="epoch",            # 每个 epoch 保存
        save_total_limit=2,
        logging_dir=os.path.join(save_path, "logs"),
        report_to="none"
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

    # === 合并 LoRA & 保存 ===
    model = model.merge_and_unload()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ 训练完成，模型已保存至：{save_path}")

if __name__ == "__main__":
    main()
