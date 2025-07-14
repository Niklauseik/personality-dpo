import os
import pandas as pd

# ========= 模型文件夹名 =========
models = {
    "base": "原始基座模型",
    "f":    "F性格模型",
    "t":    "T性格模型",
}

# ========= 数据集配置 =========
datasets = [
    {
        "name":       "imdb_sentiment",
        "file":       "imdb_sentiment_results.csv",
        "pred_col":   "prediction",
        "merge_key":  "text",
        "base_path":  "results/imdb_sentiment",
        "label_suffix": ".invalid.labeled.test.csv"
    },
    {
        "name":       "mental_sentiment",
        "file":       "mental_sentiment_results.csv",
        "pred_col":   "prediction",
        "merge_key":  "text",
        "base_path":  "results/medical",
        "label_suffix": ".invalid.labeled.test.csv"
    },
    {
        "name":       "financial_sentiment",
        "file":       "news_sentiment_results.csv",
        "pred_col":   "prediction",
        "merge_key":  "text",
        "base_path":  "results/news",
        "label_suffix": ".invalid.csv"
    },
    {
        "name":       "fiqasa_sentiment",
        "file":       "fiqasa_results.csv",
        "pred_col":   "prediction",
        "merge_key":  "text",
        "base_path":  "results/finbench",
        "label_suffix": ".invalid.csv"
    },
    {
        "name":       "sst2",
        "file":       "sst2_sentiment_results.csv",
        "pred_col":   "prediction",
        "merge_key":  "text",
        "base_path":  "results/sentiment/sst2",
        "label_suffix": ".invalid.csv"
    },
]

# ========= 主处理流程 =========
for ds in datasets:
    for mkey, mfolder in models.items():
        input_file = os.path.join(ds["base_path"], mfolder, ds["file"])
        label_file = input_file.replace(".csv", ds["label_suffix"])
        output_file = input_file.replace(".csv", ".processed.csv")

        if not os.path.exists(input_file):
            print(f"⚠️ 跳过：{input_file}（原始文件缺失）")
            continue
        if not os.path.exists(label_file):
            print(f"⚠️ 跳过：{label_file}（标注文件缺失）")
            continue

        df_main = pd.read_csv(input_file)
        df_label = pd.read_csv(label_file)

        if ds["merge_key"] not in df_main.columns or ds["merge_key"] not in df_label.columns:
            print(f"❌ 跳过：{input_file}（缺少合并字段 {ds['merge_key']}）")
            continue

        if "sentiment_label" not in df_label.columns:
            print(f"❌ 跳过：{label_file}（缺少 sentiment_label 列）")
            continue

        df_label = df_label.dropna(subset=[ds["merge_key"], "sentiment_label"])
        label_map = dict(zip(df_label[ds["merge_key"]].astype(str), df_label["sentiment_label"].astype(str)))

        def update_pred(row):
            key = str(row[ds["merge_key"]])
            return label_map.get(key, row[ds["pred_col"]])

        df_main[ds["pred_col"]] = df_main.apply(update_pred, axis=1)
        df_main.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✅ 保存：{output_file}")
