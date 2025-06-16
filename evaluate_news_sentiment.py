import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# === 模型路径和模型名 ===
base_dir = "results/news"
model_names = ["原始基座模型", "F性格模型", "T性格模型"]

# === 标签映射 ===
id2label = {
    0: "bearish",
    1: "bullish",
    2: "neutral"
}

# === 结果容器 ===
sentiment_results = {}

# === 评估逻辑 ===
for model in model_names:
    path = os.path.join(base_dir, model, "news_sentiment_results.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)

    def norm(x):
        x = str(x).strip().strip(".").lower()
        if x in ["bearish", "bullish", "neutral"]:
            return x
        for cand in ["bearish", "bullish", "neutral"]:
            if cand in x:
                return cand
        return "__invalid__"

    y_true = [id2label.get(i, "__invalid__") for i in df["label"]]
    y_pred = [norm(p) for p in df["prediction"]]
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p != "__invalid__"]
    y_true, y_pred = zip(*valid) if valid else ([], [])

    sentiment_results[model] = {
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "f1": f1_score(y_true, y_pred, average="macro") if y_true else 0.0,
        "count": len(df),
        "invalid_preds": sum(1 for p in df["prediction"].map(norm) if p == "__invalid__")
    }

# === 写入结果 ===
lines = []
lines.append("===== News Sentiment 分类任务 =====")
for model, scores in sentiment_results.items():
    lines.append(f"\n📌 Model: {model}")
    for k, v in scores.items():
        lines.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    lines.append("-" * 40)

with open(os.path.join(base_dir, "news_sentiment_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ News Sentiment 分类评估完成。")
