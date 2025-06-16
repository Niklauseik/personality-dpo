import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

# === 三个数据集配置（路径 + 文件名）===
datasets = [
    {
        "name": "imdb",
        "result_file": "imdb_sentiment_results.csv"
    },
    {
        "name": "imdb_sklearn",
        "result_file": "imdb_sklearn_sentiment_results.csv"
    },
    {
        "name": "sst2",
        "result_file": "sst2_sentiment_results.csv"
    }
]

# === 模型名子目录 ===
models = ["原始基座模型", "F性格模型", "T性格模型"]

def map_sentiment_label(text):
    if not isinstance(text, str):
        return None
    # 小写处理 + 去标点（保留 only 字母）
    text = re.sub(r'[^a-z]', '', text.lower())
    if "positive" in text:
        return 1
    if "negative" in text:
        return 0
    return None

# === 主评估流程 ===
for dataset in datasets:
    dataset_dir = os.path.join("results", "sentiment", dataset["name"])
    result_file = dataset["result_file"]
    output_path = os.path.join(dataset_dir, f"{dataset['name']}_metrics.txt")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for model in models:
            model_path = os.path.join(dataset_dir, model, result_file)
            if not os.path.exists(model_path):
                print(f"❌ 缺失结果文件：{model_path}")
                continue

            df = pd.read_csv(model_path)
            if "label" not in df.columns or "prediction" not in df.columns:
                print(f"⚠️ 无效格式：{model_path}")
                continue

            gold_labels = df["label"].tolist()
            predictions_raw = df["prediction"].tolist()
            predictions = [map_sentiment_label(p) for p in predictions_raw]

            valid_pairs = [(y, p) for y, p in zip(gold_labels, predictions) if p is not None]

            if not valid_pairs:
                f_out.write(f"\n📌 Model: {model}\n")
                f_out.write("accuracy: nan\nprecision: 0.0\nrecall: 0.0\nf1: 0.0\ncount: 0\n")
                f_out.write("-" * 40 + "\n")
                continue

            y_true, y_pred = zip(*valid_pairs)
            metrics = {
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "precision": round(precision_score(y_true, y_pred), 4),
                "recall": round(recall_score(y_true, y_pred), 4),
                "f1": round(f1_score(y_true, y_pred), 4),
                "count": len(y_true)
            }

            f_out.write(f"\n📌 Model: {model}\n")
            for k, v in metrics.items():
                f_out.write(f"{k}: {v}\n")
            f_out.write("-" * 40 + "\n")

    print(f"✅ 已评估：{dataset['name']}，结果写入 → {output_path}")
