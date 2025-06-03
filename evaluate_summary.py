import os
import pandas as pd
from evaluate import load

# === 加载评估器 ===
rouge = load("rouge")
bertscore = load("bertscore")

# === 配置两个摘要任务 ===
datasets = [
    {
        "name": "edtsum",
        "base_dir": "results/finbench",
        "filename": "edtsum_results.csv",
        "reference_col": "answer"
    },
    {
        "name": "movie_summary",
        "base_dir": "results/movie_summary",
        "filename": "movie_summary_results.csv",
        "reference_col": "PlotSummary"
    }
]

models = ["原始基座模型", "F性格模型", "T性格模型"]

# === 主流程 ===
for dataset in datasets:
    results = []
    print(f"\n📂 正在评估数据集：{dataset['name']}")

    for model in models:
        path = os.path.join(dataset["base_dir"], model, dataset["filename"])
        if not os.path.exists(path):
            print(f"❌ 缺失文件：{path}")
            continue

        df = pd.read_csv(path)
        predictions = df["prediction"].astype(str).tolist()
        references = df[dataset["reference_col"]].astype(str).tolist()

        # === 计算 ROUGE ===
        rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        rouge_scores = {
            "rouge1": round(rouge_result["rouge1"], 4),
            "rouge2": round(rouge_result["rouge2"], 4),
            "rougeL": round(rouge_result["rougeL"], 4),
            "rougeLsum": round(rouge_result["rougeLsum"], 4)
        }

        # === 计算 BERTScore ===
        bert = bertscore.compute(predictions=predictions, references=references, lang="en")
        bert_result = {
            "bert_precision": round(sum(bert['precision']) / len(bert['precision']), 4),
            "bert_recall": round(sum(bert['recall']) / len(bert['recall']), 4),
            "bert_f1": round(sum(bert['f1']) / len(bert['f1']), 4)
        }

        results.append({"model": model, **rouge_scores, **bert_result})

    # === 保存评估结果 ===
    out_path = os.path.join(dataset["base_dir"], f"{dataset['name']}_metrics.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for m in results:
            f.write(f"\n📌 Model: {m['model']}\n")
            for k, v in m.items():
                if k != "model":
                    f.write(f"{k}: {v}\n")
            f.write("-" * 40 + "\n")

    print(f"✅ 已保存至：{out_path}")
