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


# === 主流程：遍历模型 ===
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

    # === 📌 1. 医疗摘要任务 MeQSum ===
    df_meq = pd.read_excel("datasets/medical/MeQSum.xlsx")  # 读取 Excel 文件
    results_meq = []
    for _, row in tqdm(df_meq.iterrows(), total=len(df_meq), desc="MeQSum"):
        chq = str(row["CHQ"]).strip()
        prompt = f"""The following is a detailed message from a patient regarding their medical concerns. Your task is to generate a short and informative one-sentence summary that captures the core question or intent.

Message: {chq}

Summary:"""
        try:
            response = local_generate(prompt, tokenizer, model)
        except Exception as e:
            response = f"[Error] {e}"
        results_meq.append(response)

    df_meq_result = df_meq.copy()
    df_meq_result["prediction"] = results_meq
    save_dir = os.path.join("results", "medical", model_name)
    os.makedirs(save_dir, exist_ok=True)
    df_meq_result.to_csv(os.path.join(save_dir, "meqsum_results.csv"), index=False)
    print(f"✅ MeQSum 保存完成：{save_dir}")

    # === 📌 2. Mental Health Sentiment 分类 ===
    df_mental = pd.read_csv("datasets/medical/mental_health_sentiment.csv")
    results_mental = []
    for _, row in tqdm(df_mental.iterrows(), total=len(df_mental), desc="MentalSentiment"):
        post = str(row["text"]).strip()
        prompt = f"""You are given a short social media post that may reflect the mental state of the writer. 
        Please classify it as either Normal or Depression based on the emotional content.

Text: {post}

Respond with a single word: Normal or Depression."""
        try:
            response = local_generate(prompt, tokenizer, model)
        except Exception as e:
            response = f"[Error] {e}"
        results_mental.append(response)

    df_mental_result = df_mental.copy()
    df_mental_result["prediction"] = results_mental
    df_mental_result.to_csv(os.path.join(save_dir, "mental_sentiment_results.csv"), index=False)
    print(f"✅ Mental Sentiment 保存完成：{save_dir}")
