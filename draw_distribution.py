#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 dist_all 画图：
- 每个数据集两张图：counts_*.png（绝对值）、ratio_*.png（百分比）
- 一张图只画一个数据集，符合平台要求
- 使用 matplotlib，未显式指定颜色
"""

import os
import matplotlib.pyplot as plt

# ---------- 如果与统计脚本放一起，dist_all 已经存在 ----------
# 若单独运行，可先读取你保存的统计结果（dict / json / csv 都行）
# 这里假设 dist_all 变量结构：
# {
#   "imdb_sentiment": {
#        "negative": {"true": 12500, "base": 12791, "f": 11316, "t": 13250},
#        "positive": {...}
#   },
#   ...
# }

# === 以下演示直接粘 dist_all 示例；实际请用你的变量替换 ===
dist_all = {
    "imdb_sentiment": {
        "negative": {"true": 12500, "base": 12791, "f": 11316, "t": 13250},
        "positive": {"true": 12500, "base": 11421, "f": 10810, "t": 10541},
    },
    "mental_sentiment": {
        "depression": {"true": 15404, "base": 26887, "f": 27845, "t": 27229},
        "normal":     {"true": 16343, "base": 4437,  "f": 3569,  "t": 4304 },
    },
    "financial_sentiment": {
        "bearish": {"true": 1789, "base": 2840, "f": 2732, "t": 2503},
        "bullish": {"true": 2398, "base": 5226, "f": 5778, "t": 4386},
        "neutral": {"true": 7744, "base": 3863, "f": 3415, "t": 5041},
    },
    "fiqasa_sentiment": {
        "negative": {"true": 363, "base": 587, "f": 508, "t": 550},
        "neutral":  {"true":  91, "base": 340, "f": 218, "t": 512},
        "positive": {"true": 719, "base": 236, "f": 427, "t": 108},
    },
    "imdb_sklearn": {
        "negative": {"true": 4972, "base": 5316, "f": 5157, "t": 5807},
        "positive": {"true": 5028, "base": 4684, "f": 4843, "t": 4193},
    },
    "sst2": {
        "negative": {"true": 4471, "base": 5579, "f": 4887, "t": 6119},
        "positive": {"true": 5529, "base": 3857, "f": 4589, "t": 2519},
    }
}

# ---------- 绘图函数 ----------
def plot_dataset(name, data_dict, save_dir="."):
    """data_dict: {label -> {true, base, f, t}}"""
    labels = sorted(data_dict.keys())
    n_labels = len(labels)
    x = range(n_labels)

    # 收集 4 组数值
    true_vals = [data_dict[l]["true"] for l in labels]
    base_vals = [data_dict[l]["base"] for l in labels]
    f_vals    = [data_dict[l]["f"]    for l in labels]
    t_vals    = [data_dict[l]["t"]    for l in labels]

    # ---- 1. 绝对计数图 ----
    width = 0.2
    plt.figure(figsize=(1.2*n_labels + 2, 6))
    plt.bar([i - 1.5*width for i in x], true_vals, width, label="True")
    plt.bar([i - 0.5*width for i in x], base_vals, width, label="Base")
    plt.bar([i + 0.5*width for i in x], f_vals,    width, label="F model")
    plt.bar([i + 1.5*width for i in x], t_vals,    width, label="T model")

    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Sample Count")
    plt.title(f"{name} – Label Count Distribution")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"counts_{name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ 保存 {out_path}")

    # ---- 2. 百分比图 ----
    base_tot = sum(base_vals) or 1
    f_tot    = sum(f_vals) or 1
    t_tot    = sum(t_vals) or 1
    true_tot = sum(true_vals) or 1

    true_ratios = [v/true_tot for v in true_vals]
    base_ratios = [v/base_tot for v in base_vals]
    f_ratios    = [v/f_tot    for v in f_vals]
    t_ratios    = [v/t_tot    for v in t_vals]

    plt.figure(figsize=(1.2*n_labels + 2, 6))
    plt.bar([i - 1.5*width for i in x], true_ratios, width, label="True")
    plt.bar([i - 0.5*width for i in x], base_ratios, width, label="Base")
    plt.bar([i + 0.5*width for i in x], f_ratios,    width, label="F model")
    plt.bar([i + 1.5*width for i in x], t_ratios,    width, label="T model")

    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Proportion")
    plt.title(f"{name} – Label Proportion Distribution")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"ratio_{name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ 保存 {out_path}")


# ---------- 主流程 ----------
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for dname, ddata in dist_all.items():
    plot_dataset(dname, ddata, save_dir=output_dir)

print("\n🎉 所有图已生成，位于 ./plots 目录")
