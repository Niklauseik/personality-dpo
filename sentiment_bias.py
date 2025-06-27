#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 results 目录读取各模型预测文件，计算 Bias(label) = P_pred − P_true，
为每个数据集输出一张 Bias 热图 PNG（行：标签，列：模型）。
依赖：pandas, numpy, matplotlib
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# ========= 数据集配置 =========
datasets = [
    {"name": "imdb_sentiment",
     "file": "imdb_sentiment_results.csv",
     "label_map": {"0": "negative", "1": "positive"},
     "allowed_labels": None,
     "label_col": "label",
     "pred_col": "prediction",
     "base_path": "results/imdb_sentiment"},
    {"name": "mental_sentiment",
     "file": "mental_sentiment_results.csv",
     "label_map": None,
     "allowed_labels": ["normal", "depression"],
     "label_col": "label",
     "pred_col": "prediction",
     "base_path": "results/medical"},
    {"name": "financial_sentiment",
     "file": "news_sentiment_results.csv",
     "label_map": {"0": "bearish", "1": "bullish", "2": "neutral"},
     "allowed_labels": None,
     "label_col": "label",
     "pred_col": "prediction",
     "base_path": "results/news"},
    {"name": "fiqasa_sentiment",
     "file": "fiqasa_results.csv",
     "label_map": None,
     "allowed_labels": ["negative", "positive", "neutral"],
     "label_col": "answer",
     "pred_col": "prediction",
     "base_path": "results/finbench"},
    {"name": "imdb_sklearn",
     "file": "imdb_sklearn_sentiment_results.csv",
     "label_map": {"0": "negative", "1": "positive"},
     "allowed_labels": None,
     "label_col": "label",
     "pred_col": "prediction",
     "base_path": "results/sentiment/imdb_sklearn"},
    {"name": "sst2",
     "file": "sst2_sentiment_results.csv",
     "label_map": {"0": "negative", "1": "positive"},
     "allowed_labels": None,
     "label_col": "label",
     "pred_col": "prediction",
     "base_path": "results/sentiment/sst2"},
]

# ========= 模型文件夹名 =========
models = {"base": "原始基座模型",
          "f":    "F性格模型",
          "t":    "T性格模型"}

# ========= 清洗函数 =========
clean = lambda s: re.sub(r'[^a-z]', '', str(s).strip().lower())

# ========= 收集计数 =========
dist_all = defaultdict(lambda: defaultdict(lambda: {"true": 0, "base": 0, "f": 0, "t": 0}))

for ds in datasets:
    allowed = set(map(clean, ds["allowed_labels"])) if ds["allowed_labels"] else set()
    true_done = False

    for mkey, mfolder in models.items():
        fpath = os.path.join(ds["base_path"], mfolder, ds["file"])
        if not os.path.exists(fpath):
            print(f"⚠️  缺少文件：{fpath}")
            continue

        df = pd.read_csv(fpath)

        if ds["label_map"] is not None:
            df[ds["label_col"]] = df[ds["label_col"]].astype(str).map(ds["label_map"])

        df[ds["label_col"]] = df[ds["label_col"]].apply(clean)
        df[ds["pred_col"]]  = df[ds["pred_col"]].apply(clean)

        if not allowed:
            allowed = (set(map(clean, ds["label_map"].values()))
                       if ds["label_map"] else set(df[ds["label_col"]].unique()))

        if not true_done:
            for lbl, cnt in Counter(df[ds["label_col"]]).items():
                if lbl in allowed:
                    dist_all[ds["name"]][lbl]["true"] = cnt
            true_done = True

        for lbl, cnt in Counter(df[ds["pred_col"]]).items():
            if lbl in allowed:
                dist_all[ds["name"]][lbl][mkey] = cnt

# ========= 绘制热图 =========
os.makedirs("plots_bias", exist_ok=True)
for dname, lbl_dict in dist_all.items():
    labels = sorted(lbl_dict.keys())
    true_cnt = np.array([lbl_dict[l]["true"] for l in labels], dtype=float)
    true_ratio = true_cnt / true_cnt.sum()

    pred_ratio = np.stack([
        np.array([lbl_dict[l][m] for l in labels], dtype=float) /
        max(1, sum(lbl_dict[l][m] for l in labels))
        for m in models                    # 顺序保持 base, f, t
    ])
    bias = pred_ratio - true_ratio[None, :]

    plt.figure(figsize=(4 + 0.4*len(models), 2.5 + 0.35*len(labels)))
    im = plt.imshow(bias.T, aspect="auto")            # 默认配色
    plt.colorbar(im, label="Bias (Pred − True)")
    plt.xticks(range(len(models)), ["Base", "F model", "T model"])
    plt.yticks(range(len(labels)), labels)
    plt.title(f"{dname} – Bias heatmap")
    plt.tight_layout()
    plt.savefig(f"plots_bias/bias_{dname}.png", dpi=300)
    plt.close()

print("🎉 所有 Bias 热图已保存至 plots_bias/")
