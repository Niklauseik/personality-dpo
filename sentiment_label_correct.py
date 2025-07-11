import os
import pandas as pd
import time
import re
from openai import OpenAI

# === OpenAI 客户端配置 ===
client = OpenAI(api_key="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")  # ← 替换为你的 Key
MODEL = "gpt-4o-mini"

# === 文件路径配置（results 同级目录下） ===
base_dir = os.path.join(os.path.dirname(__file__), "results", "imdb_sentiment")
model_dirs = ["F性格模型", "T性格模型", "原始基座模型"]

# === 判断是否可直接复制（标准标签） ===
def is_direct_label(pred):
    cleaned = re.sub(r"[^a-z]", "", pred.lower().strip())
    return cleaned in ["mixed", "neutral"]

# === GPT 调用分类函数 ===
def classify_sentiment_gpt(pred_text):
    prompt = (
        f"You are a sentiment classification expert. "
        f"Given a sentiment expression such as \"{pred_text}\", classify it into one of the three categories: "
        f"positive, negative, or mixed. Return only one word."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Only respond with one word: positive, negative, or mixed."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        label = response.choices[0].message.content.strip().lower()
        return label if label in ["positive", "negative", "mixed"] else "mixed"
    except Exception as e:
        print(f"❌ GPT error on \"{pred_text}\": {e}")
        return "mixed"

# === 主处理流程 ===
for model_dir in model_dirs:
    folder = os.path.join(base_dir, model_dir)
    input_file = os.path.join(folder, "imdb_sentiment_results.invalid.csv")
    output_file = os.path.join(folder, "imdb_sentiment_results.invalid.labeled.test.csv")

    if not os.path.exists(input_file):
        print(f"⚠️ File not found: {input_file}")
        continue

    print(f"\n📄 Processing {model_dir} (First 3 rows)")
    df = pd.read_csv(input_file)
    df_test = df.copy()

    sentiment_labels = []
    for i, row in df_test.iterrows():
        pred = str(row.get("prediction", "")).strip()
        if is_direct_label(pred):
            label = re.sub(r"[^a-z]", "", pred.lower())
            print(f"   [{i+1}] {pred} → {label}  ✅ 复制")
        else:
            label = classify_sentiment_gpt(pred)
            print(f"   [{i+1}] {pred} → {label}  🤖 模型判断")
            time.sleep(0.3)
        sentiment_labels.append(label)

    df_test["sentiment_label"] = sentiment_labels
    df_test.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ Saved to: {output_file}")
