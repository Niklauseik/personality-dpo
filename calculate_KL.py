import numpy as np
import pandas as pd

# 改进版 KL 散度函数（log₂，单位 bits）
def kl_divergence_bits(q_counts, p_counts, epsilon=1e-12):
    p_probs = np.array(p_counts, dtype=float)
    q_probs = np.array(q_counts, dtype=float)

    # 归一化
    p_probs = p_probs / p_probs.sum()
    q_probs = q_probs / q_probs.sum()

    # 仅对 0 项加平滑，不再重新归一化
    p_probs = np.where(p_probs == 0, epsilon, p_probs)
    q_probs = np.where(q_probs == 0, epsilon, q_probs)

    return np.sum(q_probs * np.log2(q_probs / p_probs))

# 数据集定义
datasets = {
    'imdb_sentiment': {
        'base': [516, 12803, 240, 11441],
        'F':    [1325, 11775, 74, 11826],
        'T':    [244, 13253, 960, 10543],
    },
    'mental_sentiment': {
        'base': [27041, 261, 4445],
        'F':    [27934, 238, 3575],
        'T':    [27317, 121, 4309],
    },
    'financial_sentiment': {
        'base': [2840, 5226, 2, 3863],
        'F':    [2732, 5779, 5, 3415],
        'T':    [2503, 4386, 1, 5041],
    },
    'fiqasa_sentiment': {
        'base': [0, 587, 348, 238],
        'F':    [4, 511, 226, 432],
        'T':    [0, 550, 515, 108],
    },
    'imdb_sklearn': {
        'base': [5316, 4684],
        'F':    [5157, 4843],
        'T':    [5807, 4193],
    },
    'sst2': {
        'base': [2, 1, 5579, 561, 3857],
        'F':    [4, 2, 4898, 506, 4590],
        'T':    [2, 0, 6118, 1361, 2519],
    }
}

# 计算 KL 散度
results = []
for name, data in datasets.items():
    f_kl = kl_divergence_bits(data['F'], data['base'])
    t_kl = kl_divergence_bits(data['T'], data['base'])
    results.append({
        "数据集": name,
        "F模型 KL散度 (bits)": round(f_kl, 6),
        "T模型 KL散度 (bits)": round(t_kl, 6)
    })

df = pd.DataFrame(results).sort_values("数据集").reset_index(drop=True)

# 输出结果
print(df.to_string(index=False, float_format='%.6f'))
