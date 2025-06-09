import os
import pandas as pd
from datasets import load_dataset

# === 创建保存路径 ===
save_dir = "datasets/news"
os.makedirs(save_dir, exist_ok=True)

# === 1. Twitter Financial News Sentiment ===
print("📥 正在加载 Twitter Financial News Sentiment...")
ds_sentiment = load_dataset("zeroshot/twitter-financial-news-sentiment")

# 合并 train + validation
df_sentiment = pd.concat([
    ds_sentiment["train"].to_pandas(),
    ds_sentiment["validation"].to_pandas()
], ignore_index=True)

# 保存为 CSV
sentiment_path = os.path.join(save_dir, "news_sentiment.csv")
df_sentiment.to_csv(sentiment_path, index=False)
print(f"✅ Sentiment 数据已保存：{sentiment_path}")

# === 2. CNN/DailyMail Summary (test split only) ===
print("📥 正在加载 CNN/DailyMail Summary test 集...")
ds_summary = load_dataset("cnn_dailymail", "3.0.0", split="test")

# 去除 id 列
df_summary = ds_summary.remove_columns("id").to_pandas()

# 保存为 CSV
summary_path = os.path.join(save_dir, "news_summary.csv")
df_summary.to_csv(summary_path, index=False)
print(f"✅ Summary 数据已保存：{summary_path}")
