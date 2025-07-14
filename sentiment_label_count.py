import os
import re
import pandas as pd
from collections import Counter, defaultdict

# ========= 数据集配置 =========
datasets = [
    {
        "name":       "imdb_sentiment",
        "file":       "imdb_sentiment_results.processed.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/imdb_sentiment",
    },
    {
        "name":       "mental_sentiment",
        "file":       "mental_sentiment_results.processed.csv",
        "label_map":  None,
        "allowed_labels": ["normal", "depression"],
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/medical",
    },
    {
        "name":       "financial_sentiment",
        "file":       "news_sentiment_results.processed.csv",
        "label_map":  {"0": "bearish", "1": "bullish", "2": "neutral"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/news",
    },
    {
        "name":       "fiqasa_sentiment",
        "file":       "fiqasa_results.processed.csv",
        "label_map":  None,
        "allowed_labels": ["negative", "positive", "neutral"],
        "label_col":  "answer",
        "pred_col":   "prediction",
        "base_path":  "results/finbench",
    },
    {
        "name":       "imdb_sklearn",
        "file":       "imdb_sklearn_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/imdb_sklearn",
    },
    {
        "name":       "sst2",
        "file":       "sst2_sentiment_results.processed.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/sst2",
    },
]

# ========= 模型文件夹名 =========
models = {
    "base": "原始基座模型",
    "f":    "F性格模型",
    "t":    "T性格模型",
}

# ========= 文本清洗函数 =========
def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z]", "", text.strip().lower())

# ========= 自动修复函数 =========
def try_fix_csv(path, label_col, pred_col):
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    header = lines[0].strip().split(",")
    if label_col not in header or pred_col not in header:
        print(f"  ⚠️ 自动修复 CSV：尝试手动拆分 {path}")
        data = [line.strip().split(",") for line in lines[1:] if "," in line]
        if len(data) > 0:
            df = pd.DataFrame(data, columns=header)
            return df
        else:
            return pd.DataFrame()
    return None  # 无需修复

# ========= 统计容器 =========
dist_all = defaultdict(lambda: defaultdict(lambda: {"true": 0, "base": 0, "f": 0, "t": 0}))

for ds in datasets:
    print(f"🔍 处理数据集：{ds['name']}")
    allowed = set(map(clean, ds["allowed_labels"])) if ds["allowed_labels"] else set()
    true_done = False

    for mkey, mfolder in models.items():
        path = os.path.join(ds["base_path"], mfolder, ds["file"])
        if not os.path.exists(path):
            print(f"  ⚠️ 缺少文件：{path}")
            continue

        try:
            df = pd.read_csv(path, encoding="utf-8-sig", sep=",")
            df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

            # 自动修复（如果只有一列）
            if ds["label_col"] not in df.columns or ds["pred_col"] not in df.columns:
                df_fixed = try_fix_csv(path, ds["label_col"], ds["pred_col"])
                if df_fixed is not None and not df_fixed.empty:
                    df = df_fixed
                else:
                    print(f"  ❌ 无法修复或缺少关键列：{ds['label_col']} / {ds['pred_col']}")
                    continue

        except Exception as e:
            print(f"  ❌ 读取失败：{path}，错误信息：{e}")
            continue

        # 标签映射
        if ds["label_map"] is not None:
            df[ds["label_col"]] = df[ds["label_col"]].astype(str).map(ds["label_map"])

        # 文本清洗
        df[ds["label_col"]] = df[ds["label_col"]].astype(str).apply(clean)
        df["cleaned_pred"]  = df[ds["pred_col"]].astype(str).apply(clean)

        # 自动推导 allowed_labels（如果未显式给出）
        if not allowed:
            if ds["label_map"] is not None:
                allowed = set(map(clean, ds["label_map"].values()))
            else:
                allowed = set(df[ds["label_col"]].unique())

        # 统计真实标签（仅一次）
        if not true_done:
            for lbl, cnt in Counter(df[ds["label_col"]]).items():
                if lbl in allowed:
                    dist_all[ds["name"]][lbl]["true"] = cnt
            true_done = True

        # 统计 prediction 所有标签（无论是否在 allowed）
        for lbl, cnt in Counter(df["cleaned_pred"]).items():
            dist_all[ds["name"]][lbl][mkey] = cnt

# ========= 输出 TXT =========
outfile = "label_distribution_summary.txt"
with open(outfile, "w", encoding="utf-8") as f:
    for dname, label_dict in dist_all.items():
        f.write(f"======== {dname} ========\n")
        df_out = (
            pd.DataFrame(label_dict).T
              .fillna(0)
              .astype(int)
              .loc[:, ["true", "base", "f", "t"]]
              .rename(columns={
                  "true": "真实数量",
                  "base": "基座模型",
                  "f":    "F模型",
                  "t":    "T模型",
              })
              .sort_index()
        )
        f.write(df_out.to_string())
        f.write("\n\n")

print(f"\n🎉 标签统计完成！结果已保存到 {outfile}")
