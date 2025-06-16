from datasets import load_dataset
import pandas as pd
import os

# === 创建保存路径 ===
os.makedirs("datasets/sentiment", exist_ok=True)
os.makedirs("datasets/law", exist_ok=True)

# === IMDb: ajaykarthick/imdb-movie-reviews，只取 test 中 10k 条 ===
imdb = load_dataset("ajaykarthick/imdb-movie-reviews", split="test[:10000]")
imdb_df = pd.DataFrame(imdb)
imdb_df.rename(columns={"review": "text", "label": "label"}, inplace=True)
imdb_df.to_csv("datasets/sentiment/imdb.csv", index=False)

# === SST-2: stanfordnlp/sst2，只取 train 中 10k 条 ===
sst2 = load_dataset("stanfordnlp/sst2", split="train[:10000]")
sst2_df = pd.DataFrame(sst2)
sst2_df.rename(columns={"sentence": "text", "label": "label"}, inplace=True)
sst2_df.to_csv("datasets/sentiment/sst2.csv", index=False)

# === scikit-learn/imdb，只取 train 中 10k 条 ===
imdb2 = load_dataset("scikit-learn/imdb", split="train[:10000]")
imdb2_df = pd.DataFrame(imdb2)
imdb2_df.rename(columns={"review": "text", "sentiment": "label"}, inplace=True)
# 映射为数字标签
label_map = {"positive": 1, "negative": 0}
imdb2_df["label"] = imdb2_df["label"].map(label_map)
imdb2_df.to_csv("datasets/sentiment/imdb_sklearn.csv", index=False)

# === FiscalNote/billsum（法律摘要），只取 test 集 ===
billsum = load_dataset("billsum", split="test")
billsum_df = pd.DataFrame(billsum)
billsum_df.to_csv("datasets/law/billsum_test.csv", index=False)
