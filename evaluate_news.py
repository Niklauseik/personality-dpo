import os
import pandas as pd
from evaluate import load
from sklearn.metrics import accuracy_score, f1_score

# === 加载评估器 ===
rouge = load("rouge")
bertscore = load("bertscore")

# === 模型路径和模型名 ===
base_dir = "results/news"
model_names = ["原始基座模型", "F性格模型", "T性格模型"]

# === 结果容器 ===
summary_results = {}
sentiment_results = {}

# === Sentiment 分类任务 ===
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

    y_true = df["label"].map(norm).tolist()
    y_pred = df["prediction"].map(norm).tolist()
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p != "__invalid__"]
    y_true, y_pred = zip(*valid) if valid else ([], [])

    sentiment_results[model] = {
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "f1": f1_score(y_true, y_pred, average="macro") if y_true else 0.0,
        "count": len(df),
        "invalid_preds": sum(1 for p in df["prediction"].map(norm) if p == "__invalid__")
    }

# === Summary 摘要任务 ===
for model in model_names:
    path = os.path.join(base_dir, model, "news_summary_results.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    preds = df["prediction"].astype(str).tolist()
    refs = df["highlights"].astype(str).tolist()

    r_scores = rouge.compute(predictions=preds, references=refs)
    b_scores = bertscore.compute(predictions=preds, references=refs, lang="en")
    avg_bert = {
        "BERTScore_P": sum(b_scores["precision"]) / len(b_scores["precision"]),
        "BERTScore_R": sum(b_scores["recall"]) / len(b_scores["recall"]),
        "BERTScore_F1": sum(b_scores["f1"]) / len(b_scores["f1"]),
    }
    summary_results[model] = {**r_scores, **avg_bert}

# === 保存评估结果 ===
lines = []
lines.append("===== News Sentiment 分类任务 =====")
for model, scores in sentiment_results.items():
    lines.append(f"\n📌 Model: {model}")
    for k, v in scores.items():
        lines.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    lines.append("-" * 40)

lines.append("\n===== News Summary 摘要任务 =====")
for model, scores in summary_results.items():
    lines.append(f"\n📌 Model: {model}")
    for k, v in scores.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("-" * 40)

os.makedirs(base_dir, exist_ok=True)
with open(os.path.join(base_dir, "news_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n✅ 所有 News 任务评估完成，结果保存至：{os.path.join(base_dir, 'news_metrics.txt')}")
