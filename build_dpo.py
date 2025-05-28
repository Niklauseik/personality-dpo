import os
import json
import pandas as pd

# ✅ MBTI四维及其对应文件名
MBTI_DIMENSIONS = {
    0: ("E", "I", "energy_extraversion", "energy_introversion"),
    1: ("N", "S", "information_intuition", "information_sensing"),
    2: ("T", "F", "decision_thinking", "decision_feeling"),
    3: ("J", "P", "execution_judging", "execution_perceiving")
}

RAW_DIR = "datasets/training_raw"
OUTPUT_DIR = "datasets/dpo_converted"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_dpo_csv_for_dimension(dim_id):
    label_a, label_b, file_a, file_b = MBTI_DIMENSIONS[dim_id]
    path_a = os.path.join(RAW_DIR, f"en_{file_a}.json")
    path_b = os.path.join(RAW_DIR, f"en_{file_b}.json")

    with open(path_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    count = min(len(data_a), len(data_b))

    records_a = []  # file_a 模型偏好：A 是 chosen
    records_b = []  # file_b 模型偏好：B 是 chosen

    for i in range(count):
        inst_a = data_a[i].get("instruction", "").strip()
        inst_b = data_b[i].get("instruction", "").strip()
        out_a = data_a[i].get("output", "").strip()
        out_b = data_b[i].get("output", "").strip()

        if not inst_a or not out_a or not out_b:
            continue

        # file_a 偏好（如 decision_thinking）：A 是 chosen，B 是 rejected
        records_a.append({
            "chosen": json.dumps([
                {"role": "user", "content": inst_a},
                {"role": "assistant", "content": out_a}
            ], ensure_ascii=False),
            "rejected": json.dumps([
                {"role": "user", "content": inst_a},
                {"role": "assistant", "content": out_b}
            ], ensure_ascii=False),
            "score_chosen": 8,
            "score_rejected": 1
        })

        # file_b 偏好（如 decision_feeling）：B 是 chosen，A 是 rejected
        records_b.append({
            "chosen": json.dumps([
                {"role": "user", "content": inst_b if inst_b else inst_a},
                {"role": "assistant", "content": out_b}
            ], ensure_ascii=False),
            "rejected": json.dumps([
                {"role": "user", "content": inst_b if inst_b else inst_a},
                {"role": "assistant", "content": out_a}
            ], ensure_ascii=False),
            "score_chosen": 8,
            "score_rejected": 1
        })

    df_a = pd.DataFrame(records_a)
    df_b = pd.DataFrame(records_b)

    out_path_a = os.path.join(OUTPUT_DIR, f"{file_a}_dpo.csv")
    out_path_b = os.path.join(OUTPUT_DIR, f"{file_b}_dpo.csv")

    df_a.to_csv(out_path_a, index=False, encoding="utf-8-sig")
    df_b.to_csv(out_path_b, index=False, encoding="utf-8-sig")

    print(f"✅ {file_a} 样本：{len(df_a)} 条 → {out_path_a}")
    print(f"✅ {file_b} 样本：{len(df_b)} 条 → {out_path_b}")

if __name__ == "__main__":
    build_dpo_csv_for_dimension(dim_id=2)