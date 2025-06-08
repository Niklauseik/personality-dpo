import os
import pandas as pd

base_dir = "results/medical"
model_names = ["原始基座模型", "F性格模型", "T性格模型"]

def normalize(x):
    return str(x).strip().strip(".").lower()

for model in model_names:
    path = os.path.join(base_dir, model, "mental_sentiment_results.csv")
    if not os.path.exists(path):
        print(f"❌ 文件不存在：{path}")
        continue

    df = pd.read_csv(path)
    labels = df["label"].map(normalize).tolist()
    preds = df["prediction"].map(normalize).tolist()

    label_set = sorted(set(labels))
    pred_set = sorted(set(preds))

    print(f"\n📌 模型：{model}")
    print(f"✔️ 标签中的类别（label）: {label_set}")
    print(f"✔️ 预测中的类别（prediction）: {pred_set}")
