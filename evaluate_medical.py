import os
import pandas as pd
from evaluate import load
from sklearn.metrics import accuracy_score, f1_score

# === 加载评估器 ===
rouge = load("rouge")
bertscore = load("bertscore")

# === 模型路径配置 ===
base_dir = "results/medical"
model_names = ["原始基座模型", "F性格模型", "T性格模型"]

summary_results = {}
sentiment_results = {}

# === 常用函数 ===
def normalize(x):
    return str(x).strip().strip(".").lower()

valid_labels = {"normal", "depression"}

def resolve_prediction(raw_pred, gold_label):
    pred = normalize(raw_pred)
    if pred in valid_labels:
        return pred
    elif "depression" in pred:
        return "depression"
    elif "normal" in pred:
        return "normal"
    else:
        # 无法识别 → 强制判错
        return "normal" if gold_label == "depression" else "depression"

# === 摘要任务（MeQSum）===
for model in model_names:
    path = os.path.join(base_dir, model, "meqsum_results.csv")
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)
    preds = df["prediction"].astype(str).tolist()
    refs = df["Summary"].astype(str).tolist()

    r_scores = rouge.compute(predictions=preds, references=refs)
    b_scores = bertscore.compute(predictions=preds, references=refs, lang="en")
    avg_bert = {
        "BERTScore_P": sum(b_scores["precision"]) / len(b_scores["precision"]),
        "BERTScore_R": sum(b_scores["recall"]) / len(b_scores["recall"]),
        "BERTScore_F1": sum(b_scores["f1"]) / len(b_scores["f1"]),
    }

    summary_results[model] = {**r_scores, **avg_bert}

# === 情感分类任务（Mental Sentiment）===
for model in model_names:
    path = os.path.join(base_dir, model, "mental_sentiment_results.csv")
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)
    raw_labels = df["label"].map(normalize).tolist()
    raw_preds = df["prediction"].astype(str).tolist()

    resolved_preds = []
    for pred, label in zip(raw_preds, raw_labels):
        resolved_preds.append(resolve_prediction(pred, label))

    acc = accuracy_score(raw_labels, resolved_preds)
    f1 = f1_score(raw_labels, resolved_preds, pos_label="depression", average="binary")

    sentiment_results[model] = {
        "accuracy": acc,
        "f1": f1,
        "count": len(raw_labels)
    }

# === 写入 metrics.txt ===
lines = []

lines.append("===== Mental Sentiment 分类任务 =====")
for model, scores in sentiment_results.items():
    lines.append(f"\n📌 Model: {model}")
    for k, v in scores.items():
        lines.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    lines.append("-" * 40)

lines.append("\n===== MeQSum 摘要任务 =====")
for model, scores in summary_results.items():
    lines.append(f"\n📌 Model: {model}")
    for k, v in scores.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("-" * 40)

os.makedirs(base_dir, exist_ok=True)
with open(os.path.join(base_dir, "metrics.txt"), "w") as f:
    f.write("\n".join(lines))
