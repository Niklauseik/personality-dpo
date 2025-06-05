import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === 配置路径与模型 ===
base_dir = "results/imdb_sentiment"
models = ["原始基座模型", "F性格模型", "T性格模型"]
result_file = "imdb_sentiment_results.csv"

# === 映射模型输出为 0 或 1 ===
def map_sentiment_label(text):
    if not isinstance(text, str):
        return None
    text = text.lower()
    if "positive" in text:
        return 1
    if "negative" in text:
        return 0
    return None  # 其他类别或无法识别的返回 None

# === 主流程 ===
output_path = os.path.join(base_dir, "imdb_sentiment_metrics.txt")
with open(output_path, "w", encoding="utf-8") as f_out:
    for model in models:
        path = os.path.join(base_dir, model, result_file)
        if not os.path.exists(path):
            print(f"❌ 缺失结果文件：{path}")
            continue

        df = pd.read_csv(path)
        gold_labels = df["label"].tolist()
        predictions_raw = df["prediction"].tolist()

        predictions = [map_sentiment_label(p) for p in predictions_raw]
        valid = [(y, p) for y, p in zip(gold_labels, predictions) if p is not None]

        if not valid:
            f_out.write(f"\n📌 Model: {model}\n")
            f_out.write("accuracy: nan\nprecision: 0.0\nrecall: 0.0\nf1: 0.0\ncount: 0\n")
            f_out.write("-" * 40 + "\n")
            continue

        y_true, y_pred = zip(*valid)

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

print(f"\n✅ 所有 IMDb 情感分类评估完成，结果保存至：{output_path}")
