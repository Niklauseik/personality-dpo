import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === 根目录，确保是在 personality/ 下运行 ===
base_path = "./results/benchmark"

# === 模型文件夹名称 ===
model_folders = ["F性格模型", "T性格模型", "原始基座模型"]

# === 数据集文件名映射（新增 GSM8K） ===
files = {
    "ARC (easy)": "arc_easy_test800_results.csv",
    "BoolQ": "boolq_train800_results.csv",
    "GSM8K": "gsm8k_test800_results.csv"
}

# === 提取函数 ===
def extract_upper_letter(text):
    match = re.search(r'\b([A-D])\b', str(text).upper())
    return match.group(1) if match else None

def extract_bool(text):
    if isinstance(text, str):
        text_lower = text.lower()
        if 'true' in text_lower:
            return True
        elif 'false' in text_lower:
            return False
    elif isinstance(text, bool):
        return text
    return None

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

# === 收集所有结果 ===
all_results = []

for model_name in model_folders:
    model_path = os.path.join(base_path, model_name)

    for dataset_name, filename in files.items():
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        if dataset_name == "ARC (easy)":
            df["label_clean"] = df["label"].apply(extract_upper_letter)
            df["prediction_clean"] = df["prediction"].apply(extract_upper_letter)
        elif dataset_name == "BoolQ":
            df["label_clean"] = df["label"].apply(extract_bool)
            df["prediction_clean"] = df["prediction"].apply(extract_bool)
        elif dataset_name == "GSM8K":
            df["label_clean"] = df["label"].astype(str).str.strip()
            df["prediction_clean"] = df["prediction"].astype(str).str.strip()

        df_valid = df.dropna(subset=["label_clean", "prediction_clean"])

        if dataset_name == "GSM8K":
            accuracy = accuracy_score(df_valid["label_clean"], df_valid["prediction_clean"])
            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": None,
                "recall": None,
                "f1": None
            }
        else:
            metrics = compute_metrics(df_valid["label_clean"], df_valid["prediction_clean"])

        all_results.append({
            "Model": model_name,
            "Dataset": dataset_name,
            **metrics
        })

# === 输出为 DataFrame 结果表 ===
df_metrics = pd.DataFrame(all_results)
print(df_metrics)

# === 保存结果到 txt 文件 ===
output_path = os.path.join(base_path, "benchmark_metrics_summary.txt")

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df_metrics.iterrows():
        f.write(
            f"\n📌 Model: {row['Model']}\n"
            f"📊 Dataset: {row['Dataset']}\n"
            f"✅ Accuracy: {row['accuracy']}\n"
            f"✅ Precision: {row['precision']}\n"
            f"✅ Recall: {row['recall']}\n"
            f"✅ F1 Score: {row['f1']}\n"
            f"{'-'*40}\n"
        )

print(f"\n📁 已将结果保存到：{output_path}")
