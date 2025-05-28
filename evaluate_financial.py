import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 根路径
base_path = "./results/finbench"
model_folders = ["F性格模型", "T性格模型", "原始基座模型"]

# 每个数据集的评价方式（分类标签归一）
datasets = {
    "cfa_1000_results.csv": lambda x: str(x).strip().upper(),        # A/B/C
    "fiqasa_results.csv": lambda x: str(x).strip().lower(),          # positive/negative/neutral
    "german_400_results.csv": lambda x: str(x).strip().lower(),      # good/bad
    "bigdata_1400_results.csv": lambda x: str(x).strip().capitalize(),  # Rise/Fall
    "headlines_2000_results.csv": lambda x: str(x).strip().capitalize()  # Yes/No
}

# 结果容器
results = []

for model in model_folders:
    model_path = os.path.join(base_path, model)

    for file_name, clean_fn in datasets.items():
        file_path = os.path.join(model_path, file_name)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df["label_clean"] = df["answer"].apply(clean_fn)
        df["prediction_clean"] = df["prediction"].apply(clean_fn)

        # 去除无法对比的样本
        df_valid = df.dropna(subset=["label_clean", "prediction_clean"])
        y_true = df_valid["label_clean"]
        y_pred = df_valid["prediction_clean"]

        # 计算 accuracy 和 F1
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        results.append({
            "模型": model,
            "数据集": file_name.replace("_results.csv", ""),
            "Accuracy": round(acc, 4),
            "F1": round(f1, 4)
        })

# 输出结果表格
df_metrics = pd.DataFrame(results)
print(df_metrics)

# 另存为 txt
save_path = os.path.join(base_path, "finbench_metrics_summary.txt")
with open(save_path, "w", encoding="utf-8") as f:
    for _, row in df_metrics.iterrows():
        f.write(
            f"📌 模型: {row['模型']}\n"
            f"📊 数据集: {row['数据集']}\n"
            f"✅ Accuracy: {row['Accuracy']}\n"
            f"✅ F1 Score: {row['F1']}\n"
            f"{'-'*40}\n"
        )

print(f"\n✅ 已保存至：{save_path}")
