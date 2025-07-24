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

# === PROMPT 示例 ===

sentiment_prompt = """You are analyzing financial news headlines. Each headline reflects a short financial opinion or fact.
Please classify the overall sentiment into one of the following categories:
- Bearish
- Bullish
- Neutral

Respond with one word only.

Example:
Text: $GM - GM loses a bull
Answer: Bearish

Now classify the following:
"""

summary_prompt = """You are given a full news article from CNN. Your job is to summarize it into a short highlight (1–3 sentences), similar to a brief bullet point.

Example:
Article: (CNN)Five Americans who were monitored for three weeks at an Omaha, Nebraska, hospital after being exposed to Ebola in West Africa have been released, a Nebraska Medicine spokesman said in an email Wednesday. One of the five had a heart-related issue on Saturday and has been discharged but hasn't left the area, Taylor Wilson wrote. The others have already gone home. They were exposed to Ebola in Sierra Leone in March, but none developed the deadly virus. They are clinicians for Partners in Health, a Boston-based aid group. They all had contact with a colleague who was diagnosed with the disease and is being treated at the National Institutes of Health in Bethesda, Maryland. As of Monday, that health care worker is in fair condition. The Centers for Disease Control and Prevention in Atlanta has said the last of 17 patients who were being monitored are expected to be released by Thursday. More than 10,000 people have died in a West African epidemic of Ebola that dates to December 2013, according to the World Health Organization. Almost all the deaths have been in Guinea, Liberia and Sierra Leone. Ebola is spread by direct contact with the bodily fluids of an infected person.
Highlight: 17 Americans were exposed to the Ebola virus while in Sierra Leone in March. Another person was diagnosed with the disease and taken to hospital in Maryland. National Institutes of Health says the patient is in fair condition after weeks of treatment.

Now summarize the following article:
"""

# === 加载数据 ===
df_sent = pd.read_csv("datasets/news/news_sentiment.csv")
df_sum = pd.read_csv("datasets/news/news_summary.csv")

# === 遍历每个模型，运行两个任务 ===
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

    save_dir = os.path.join("results", "news", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # === 1. Sentiment 任务 ===
    sentiment_outputs = []
    for _, row in tqdm(df_sent.iterrows(), total=len(df_sent), desc="Sentiment"):
        text = str(row["text"]).strip()
        prompt = sentiment_prompt + f"\nText: {text}\nAnswer:"
        try:
            pred = local_generate(prompt, tokenizer, model)
        except Exception as e:
            pred = f"[Error] {e}"
        sentiment_outputs.append(pred)

    df_sent_result = df_sent.copy()
    df_sent_result["prediction"] = sentiment_outputs
    df_sent_result.to_csv(os.path.join(save_dir, "news_sentiment_results.csv"), index=False)
    print(f"✅ Sentiment 保存完成：{save_dir}")

    # === 2. Summary 任务 ===
    summary_outputs = []
    for _, row in tqdm(df_sum.iterrows(), total=len(df_sum), desc="Summary"):
        article = str(row["article"]).strip()
        prompt = summary_prompt + f"\nArticle: {article}\nHighlight:"
        try:
            pred = local_generate(prompt, tokenizer, model)
        except Exception as e:
            pred = f"[Error] {e}"
        summary_outputs.append(pred)

    df_sum_result = df_sum.copy()
    df_sum_result["prediction"] = summary_outputs
    df_sum_result.to_csv(os.path.join(save_dir, "news_summary_results.csv"), index=False)
    print(f"✅ Summary 保存完成：{save_dir}")
