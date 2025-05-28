import os
import json
from pathlib import Path

# === 参数配置 ===
input_dir = Path("datasets/training_raw")
output_dir = Path("datasets/dpo")
output_dir.mkdir(parents=True, exist_ok=True)

# === MBTI 四个维度及其子类 ===
dimension_pairs = {
    "decision": ("feeling", "thinking"),
    "information": ("sensing", "intuition"),
    "energy": ("extraversion", "introversion"),
    "execution": ("perceiving", "judging"),
}

# === 加载所有原始数据 ===
def load_raw_data():
    raw_data = {}
    for file in input_dir.glob("*.json"):
        parts = file.stem.split("_")  # e.g., en_information_sensing
        if len(parts) < 3:
            continue
        dimension = parts[1]
        subtype = parts[2]
        with open(file, "r", encoding="utf-8") as f:
            raw_data[(dimension, subtype)] = json.load(f)
    return raw_data

# === 构建纯文本扁平化格式的 DPO 数据集 ===
def build_dpo_dataset(dimension: str, num_samples: int, preferred_type: str):
    raw_data = load_raw_data()

    if dimension not in dimension_pairs:
        raise ValueError(f"Unknown dimension: {dimension}")

    t1, t2 = dimension_pairs[dimension]
    if preferred_type not in (t1, t2):
        raise ValueError(f"{preferred_type} must be one of ({t1}, {t2})")

    key_chosen = (dimension, preferred_type)
    key_rejected = (dimension, t2 if preferred_type == t1 else t1)

    if key_chosen not in raw_data or key_rejected not in raw_data:
        raise FileNotFoundError(f"Missing data for {key_chosen} or {key_rejected}")

    chosen_data = raw_data[key_chosen]
    rejected_data = raw_data[key_rejected]
    max_len = min(len(chosen_data), len(rejected_data), num_samples)

    dpo_data = []
    for i in range(max_len):
        dpo_data.append({
            "prompt": chosen_data[i]["instruction"],
            "chosen": chosen_data[i]["output"],
            "rejected": rejected_data[i]["output"]
        })

    output_path = output_dir / f"{dimension}_{preferred_type}_dpo_flat.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Flat DPO 数据已保存: {output_path}（共 {len(dpo_data)} 条）")

# === 主函数 ===
def main():
    dimension = "decision"
    preferred_type = "feeling"  # thinking 是 chosen，feeling 是 rejected
    num_samples = 16000

    build_dpo_dataset(dimension, num_samples, preferred_type)

if __name__ == "__main__":
    main()
