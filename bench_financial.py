import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 模型路径（base / F / T）===
model_configs = {
    "原始基座模型": "./llama-3B-Instruct",
    "F性格模型": "./dpo_outputs/model_f_3B",
    "T性格模型": "./dpo_outputs/model_t_3B"
}

# === FinBench 数据集（读取全部）===
finbench_datasets = {
    "german_400": pd.read_csv("datasets/finbench/german_700.csv"),
    "cfa_1000": pd.read_csv("datasets/finbench/cfa_1000.csv"),
    "fiqasa": pd.read_csv("datasets/finbench/fiqasa.csv"),
    "bigdata_1400": pd.read_csv("datasets/finbench/bigdata_1400.csv"),
    "headlines_2000": pd.read_csv("datasets/finbench/headlines_2000.csv")
}

# === 推理函数 ===
def local_generate(prompt, tokenizer, model):
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# === 主流程：所有模型 × 所有数据集 ===
for model_name, model_path in model_configs.items():
    print(f"\n🧪 正在测试模型：{model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    for dataset_name, df in finbench_datasets.items():
        print(f"\n📂 处理数据集：{dataset_name}")
        predictions = []

        for idx, row in df.iterrows():
            raw_text = str(row["text"]).strip()
            cleaned = re.sub(r'\bAnswer\s*[:：]?\s*$', '', raw_text, flags=re.IGNORECASE).strip()

            # === 构造 prompt ===
            if dataset_name == "german_400":
                prompt = f"{cleaned}\n\nOnly respond with one word: good or bad. For example: good"
            elif dataset_name == "cfa_1000":
                prompt = f"{cleaned}\n\nOnly respond with one word. For example: C"
            elif dataset_name == "fiqasa":
                prompt = (
                    "You are a financial sentiment classifier. "
                    "Respond with only one word: either 'positive', 'neutral', or 'negative'.\n\n"
                    f"{cleaned}"
                )
            elif dataset_name == "bigdata_1400":
                prompt = (
                    f"{cleaned}\n\nKindly confirm with only one word: Rise or Fall. For example: Rise"
                )
            elif dataset_name == "headlines_2000":
                prompt = (
                    f"{cleaned}\n\nOnly respond with one word: Yes or No. For example: No"
                )
            else:
                prompt = cleaned

            try:
                response = local_generate(prompt, tokenizer, model)
            except Exception as e:
                response = f"[Error] {e}"

            predictions.append(response)

        # 保存结果
        df_result = df.copy()
        df_result["prediction"] = predictions
        save_dir = os.path.join("results", "finbench", model_name)
        os.makedirs(save_dir, exist_ok=True)
        df_result.to_csv(os.path.join(save_dir, f"{dataset_name}_results.csv"), index=False, encoding="utf-8")
        print(f"✅ 保存完成：{dataset_name} → {save_dir}")
