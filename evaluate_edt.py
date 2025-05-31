import os
import pandas as pd
from rouge_score import rouge_scorer
from evaluate import load

# === 配置路径 ===
models = ["原始基座模型", "F性格模型", "T性格模型"]
base_dir = "results/finbench"
datafile = "edtsum_results.csv"

# === 加载 BERTScore（官方实现） ===
bertscore = load("bertscore")

# === 本地 ROUGE 计算 ===
def compute_rouge_score(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for k in scores:
            scores[k].append(result[k].fmeasure)
    return {k: round(sum(v)/len(v), 4) for k, v in scores.items()}

# === 遍历每个模型结果目录 ===
all_metrics = []

for model in models:
    path = os.path.join(base_dir, model, datafile)
    if not os.path.exists(path):
        print(f"❌ 缺失结果文件：{path}")
        continue

    df = pd.read_csv(path)
    predictions = df["prediction"].astype(str).tolist()
    references = df["answer"].astype(str).tolist()

    # === 计算 ROUGE ===
    rouge_result = compute_rouge_score(predictions, references)

    # === 计算 BERTScore ===
    bert = bertscore.compute(predictions=predictions, references=references, lang="en")
    bert_result = {
        "bert_precision": round(sum(bert['precision']) / len(bert['precision']), 4),
        "bert_recall": round(sum(bert['recall']) / len(bert['recall']), 4),
        "bert_f1": round(sum(bert['f1']) / len(bert['f1']), 4)
    }

    all_metrics.append({"model": model, **rouge_result, **bert_result})

# === 输出汇总 ===
out_path = os.path.join(base_dir, "edtsum_metrics.txt")
with open(out_path, "w", encoding="utf-8") as f:
    for m in all_metrics:
        f.write(f"\n📌 Model: {m['model']}\n")
        for k, v in m.items():
            if k != "model":
                f.write(f"{k}: {v}\n")
        f.write("-" * 40 + "\n")

print(f"\n✅ 评估完成，结果已保存：{out_path}")
