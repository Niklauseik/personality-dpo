import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 模型路径（base / F / T）===
model_configs = {
    "原始基座模型": "./llama-3B-Instruct",
    "F性格模型": "./dpo_outputs/model_f_3B",
    "T性格模型": "./dpo_outputs/model_t_3B"
}

# === 加载 EDTSum 数据集 ===
df = pd.read_csv("datasets/finbench/edtsum.csv")  # 包含 query 和 answer 两列

# === 推理函数 ===
def local_generate(prompt, tokenizer, model):
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# === 主流程：只测试 EDTSum 一份数据集 ===
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

    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = str(row["query"]).strip()
        try:
            response = local_generate(prompt, tokenizer, model)
        except Exception as e:
            response = f"[Error] {e}"
        predictions.append(response)

    df_result = df.copy()
    df_result["prediction"] = predictions

    save_dir = os.path.join("results", "finbench", model_name)
    os.makedirs(save_dir, exist_ok=True)
    df_result.to_csv(os.path.join(save_dir, "edtsum_results.csv"), index=False, encoding="utf-8")
    print(f"✅ 保存完成：edtsum → {save_dir}")
