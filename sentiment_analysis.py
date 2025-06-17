import os
import pandas as pd
import re
from sklearn.metrics import classification_report
from collections import Counter

# === 数据集配置 ===
datasets = [
    {
        "name": "imdb_sentiment",
        "file": "imdb_sentiment_results.csv",
        "label_map": {"0": "negative", "1": "positive"},
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/imdb_sentiment"
    },
    {
        "name": "mental_sentiment",
        "file": "mental_sentiment_results.csv",
        "label_map": None,
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/medical"
    },
    {
        "name": "financial_sentiment",
        "file": "news_sentiment_results.csv",
        "label_map": {"0": "bearish", "1": "bullish", "2": "neutral"},
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/news"
    },
    {
        "name": "fiqasa_sentiment",
        "file": "fiqasa_results.csv",
        "label_map": None,
        "label_col": "answer",
        "pred_col": "prediction",
        "base_path": "results/finbench"
    },
    {
        "name": "imdb_sklearn",
        "file": "imdb_sklearn_sentiment_results.csv",
        "label_map": {"0": "negative", "1": "positive"},
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/sentiment/imdb_sklearn"
    },
    {
        "name": "sst2",
        "file": "sst2_sentiment_results.csv",
        "label_map": {"0": "negative", "1": "positive"},
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/sentiment/sst2"
    }
]

# === 模型配置 ===
models = {
    "base": "原始基座模型",
    "f": "F性格模型",
    "t": "T性格模型"
}

# === 清洗字符串函数 ===
def clean(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-z]', '', text.strip().lower())

all_results = []

for ds in datasets:
    for model_key, model_folder in models.items():
        path = os.path.join(ds["base_path"], model_folder, ds["file"])
        if not os.path.exists(path):
            print(f"❌ 文件不存在：{path}")
            continue

        df = pd.read_csv(path)
        label_col = ds["label_col"]
        pred_col = ds["pred_col"]
        label_map = ds["label_map"]

        if label_map:
            df[label_col] = df[label_col].astype(str).map(label_map)

        df[pred_col] = df[pred_col].astype(str).apply(clean)
        df[label_col] = df[label_col].astype(str).apply(clean)

        valid = df[df[pred_col].isin(df[label_col].unique())]
        y_true = valid[label_col].tolist()
        y_pred = valid[pred_col].tolist()

        dist_pred = Counter(y_pred)
        dist_true = Counter(y_true)
        total_pred = sum(dist_pred.values())
        total_true = sum(dist_true.values())

        all_labels = sorted(set(y_true + y_pred))
        row = {
            "dataset": ds["name"],
            "model": model_folder
        }

        # === 分布偏移统计 ===
        for label in all_labels:
            pred_ratio = dist_pred[label] / total_pred if total_pred else 0
            true_ratio = dist_true[label] / total_true if total_true else 0
            row[f"Pred_{label}_ratio"] = round(pred_ratio, 4)
            row[f"True_{label}_ratio"] = round(true_ratio, 4)
            row[f"Bias_{label}"] = round(pred_ratio - true_ratio, 4)

        # === 分类评估指标 ===
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        for label in all_labels:
            if label in report:
                for metric in ["precision", "recall", "f1-score", "support"]:
                    row[f"{label}_{metric}"] = round(report[label][metric], 4)

        all_results.append(row)

# === 输出为 CSV ===
df_out = pd.DataFrame(all_results)

# === 按数据集分组写入不同 Sheet ===
# === 按数据集分组并转置写入 Excel ===
output_excel = "sentiment_bias_full_analysis_transposed.xlsx"
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    for dataset_name in df_out["dataset"].unique():
        df_subset = df_out[df_out["dataset"] == dataset_name]
        df_pivot = df_subset.drop(columns=["dataset"]).set_index("model").T
        df_pivot.to_excel(writer, sheet_name=dataset_name[:31])

print(f"✅ 分析结果（转置版）已写入：{output_excel}")
