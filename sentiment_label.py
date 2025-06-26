#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计各数据集合法标签的真实 / 预测分布
输出：label_distribution_summary.txt
"""

import os
import re
import pandas as pd
from collections import Counter, defaultdict

# ========= 数据集配置 =========
datasets = [
    # 1. IMDB
    {
        "name":       "imdb_sentiment",
        "file":       "imdb_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/imdb_sentiment",
    },
    # 2. Mental health (只有 normal / depression 两类)
    {
        "name":       "mental_sentiment",
        "file":       "mental_sentiment_results.csv",
        "label_map":  None,                           # 文件里本来就是英文标签
        "allowed_labels": ["normal", "depression"],   # 明确合法标签
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/medical",
    },
    # 3. Financial news (bearish / bullish / neutral)
    {
        "name":       "financial_sentiment",
        "file":       "news_sentiment_results.csv",
        "label_map":  {"0": "bearish", "1": "bullish", "2": "neutral"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/news",
    },
    # 4. FIQASA (negative / positive / neutral)
    {
        "name":       "fiqasa_sentiment",
        "file":       "fiqasa_results.csv",
        "label_map":  None,
        "allowed_labels": ["negative", "positive", "neutral"],
        "label_col":  "answer",
        "pred_col":   "prediction",
        "base_path":  "results/finbench",
    },
    # 5. IMDb (sklearn baseline)
    {
        "name":       "imdb_sklearn",
        "file":       "imdb_sklearn_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/imdb_sklearn",
    },
    # 6. SST-2
    {
        "name":       "sst2",
        "file":       "sst2_sentiment_results.csv",
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
    """保留小写英文字母，去掉空格与其他字符"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z]", "", text.strip().lower())

# ========= 统计容器 =========
# {dataset -> {label -> {true, base, f, t}}}
dist_all = defaultdict(lambda: defaultdict(lambda: {"true": 0, "base": 0, "f": 0, "t": 0}))

for ds in datasets:
    print(f"🔍 处理数据集：{ds['name']}")
    allowed = set(map(clean, ds["allowed_labels"])) if ds["allowed_labels"] else set()
    true_done = False  # 只统计一次真实标签

    for mkey, mfolder in models.items():
        path = os.path.join(ds["base_path"], mfolder, ds["file"])
        if not os.path.exists(path):
            print(f"  ⚠️ 缺少文件：{path}")
            continue

        df = pd.read_csv(path)

        # --- 映射标签 ---
        if ds["label_map"] is not None:
            df[ds["label_col"]] = df[ds["label_col"]].astype(str).map(ds["label_map"])

        # --- 清洗 ---
        df[ds["label_col"]] = df[ds["label_col"]].astype(str).apply(clean)
        df[ds["pred_col"]]  = df[ds["pred_col"]].astype(str).apply(clean)

        # 若未显式给 allowed，则用 label_map 的值；仍为空时再退化为真实标签集合
        if not allowed:
            if ds["label_map"] is not None:
                allowed = set(map(clean, ds["label_map"].values()))
            else:
                allowed = set(df[ds["label_col"]].unique())

        # --- 统计真实标签 ---
        if not true_done:
            for lbl, cnt in Counter(df[ds["label_col"]]).items():
                if lbl in allowed:
                    dist_all[ds["name"]][lbl]["true"] = cnt
            true_done = True

        # --- 统计预测标签 ---
        for lbl, cnt in Counter(df[ds["pred_col"]]).items():
            if lbl in allowed:
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
print(f"\n🎉 统计完成！结果已保存到 {outfile}")
