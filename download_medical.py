from datasets import load_dataset
import pandas as pd
import os

# === 加载数据集 ===
dataset = load_dataset("btwitssayan/sentiment-analysis-for-mental-health", split="train")

# === 筛选 Normal 和 Anxiety 样本 ===
subset = dataset.filter(lambda x: x["status"] in ["Normal", "Depression"])

# === 转换为 DataFrame 并重命名列 ===
df = pd.DataFrame(subset)
df = df[["statement", "status"]]
df.columns = ["text", "label"]

# === 创建保存目录并导出 CSV ===
os.makedirs("datasets/medical", exist_ok=True)
output_path = "datasets/medical/mental_health_sentiment.csv"
df.to_csv(output_path, index=False)

print(f"✅ 数据已保存到：{output_path}（共 {len(df)} 条样本）")
