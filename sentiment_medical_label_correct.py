import os
import pandas as pd
import time
import re
from openai import OpenAI

# === OpenAI 客户端配置 ===
client = OpenAI(api_key="")  # ← 替换为你的 Key
MODEL = "gpt-4o-mini"

# === 文件路径配置 ===
base_dir = os.path.join(os.path.dirname(__file__), "results", "medical")
model_dirs = ["F性格模型", "T性格模型", "原始基座模型"]

# === 判断 prediction 是否可直接归类 ===
def is_direct_label(pred):
    cleaned = re.sub(r"[^a-z]", "", pred.lower().strip())
    return cleaned in ["normal", "depression", "invalid"]

# === 调用 GPT 对模型 prediction 进行归类 ===
def classify_prediction_output(pred_text):
    prompt = (
        f"The following is a prediction result from a mental health model. "
        f"Please classify this output into one of three categories:\n"
        f"- normal: indicates normal mental state\n"
        f"- depression: indicates depressive state\n"
        f"- invalid: vague or evasive answer like 'I need more information'\n\n"
        f"Prediction: \"{pred_text}\"\n\n"
        f"Return only one word: normal, depression, or invalid."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Only respond with one word: normal, depression, or invalid."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        label = response.choices[0].message.content.strip().lower()
        return label if label in ["normal", "depression", "invalid"] else "invalid"
    except Exception as e:
        print(f"❌ GPT error on \"{pred_text}\": {e}")
        return "invalid"

# === 主处理流程 ===
for model_dir in model_dirs:
    folder = os.path.join(base_dir, model_dir)
    input_file = os.path.join(folder, "mental_sentiment_results.invalid.csv")
    output_file = os.path.join(folder, "mental_sentiment_results.invalid.labeled.test.csv")

    if not os.path.exists(input_file):
        print(f"⚠️ File not found: {input_file}")
        continue

    print(f"\n📄 Processing {model_dir} (full data)")
    df = pd.read_csv(input_file)
    df_test = df.copy()

    sentiment_labels = []
    for i, row in df_test.iterrows():
        pred = str(row.get("prediction", "")).strip()
        if is_direct_label(pred):
            label = re.sub(r"[^a-z]", "", pred.lower())
            print(f"   [{i+1}] {pred[:60]}... → {label}  ✅ 复制")
        else:
            label = classify_prediction_output(pred)
            print(f"   [{i+1}] {pred[:60]}... → {label}  🤖 模型判断")
            time.sleep(0.3)
        sentiment_labels.append(label)

    df_test["sentiment_label"] = sentiment_labels
    df_test.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ Saved to: {output_file}")
