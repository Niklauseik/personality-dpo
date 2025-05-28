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
from transformers.trainer_callback import TrainerCallback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# === 自定义 Callback：记录 loss 到文件 ===
class LossLoggerCallback(TrainerCallback):
    def __init__(self, save_path):
        self.log_path = os.path.join(save_path, "loss_log.txt")
        self.step_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.step_count += 1
            if self.step_count % 100 == 0:
                with open(self.log_path, "a") as f:
                    f.write(f"Step {state.global_step}: loss = {logs['loss']:.4f}\n")

def train_dpo_model(data_path: str, save_path: str):
    model_path = "./llama-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # === 参考模型（冻结 + 放到 CPU）===
    ref_model = AutoModelForCausalLM.from_pretrained(model_path).to("cpu")
    for p in ref_model.parameters():
        p.requires_grad = False

    # === 加载前 2000 条数据 ===
    train_ds = load_dataset("json", data_files={"train": data_path})["train"].select(range(2000))

    # === DPO 配置 ===
    dpo_cfg = DPOConfig(
        output_dir=save_path,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=6,
        learning_rate=2e-5,
        beta=0.5,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=os.path.join(save_path, "logs"),
        report_to="none"
    )

    # === 初始化训练器 ===
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[LossLoggerCallback(save_path)]
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
