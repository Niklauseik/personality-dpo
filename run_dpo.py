import os
import argparse
from build_dpo import build_dpo_csv_for_dimension
from dpo_training_trl import train_dpo_model

# ✅ MBTI 四维及其对应标识与名称
MBTI_DIMENSIONS = {
    0: ("E", "I", "energy_extraversion", "energy_introversion"),
    1: ("N", "S", "information_intuition", "information_sensing"),
    2: ("T", "F", "decision_thinking", "decision_feeling"),
    3: ("J", "P", "execution_judging", "execution_perceiving")
}

def main():
    print("📌 请选择要训练的 MBTI 维度：")
    for dim_id, (pos, neg, pos_name, neg_name) in MBTI_DIMENSIONS.items():
        print(f"  {dim_id}: {pos}/{neg} ({pos_name} vs. {neg_name})")

    dim_id = input("请输入维度编号（0-3）: ").strip()
    if not dim_id.isdigit() or int(dim_id) not in MBTI_DIMENSIONS:
        print("❌ 输入无效，请输入 0 到 3 之间的数字。")
        return

    dim_id = int(dim_id)
    pos, neg, pos_name, neg_name = MBTI_DIMENSIONS[dim_id]

    # ✅ 构建数据集
    print(f"📂 正在构建 {pos}/{neg} 维度的数据集...")
    build_dpo_csv_for_dimension(dim_id=dim_id)
    print("✅ 数据集构建完成。")

    # ✅ 启动训练
    print(f"🚀 开始训练 {neg} 模型...")
    train_dpo_model(
        data_path=f"./datasets/dpo_converted/{neg_name}_dpo.csv",
        save_path=f"./dpo_outputs/model_{neg}_3B"
    )

    print(f"🚀 开始训练 {pos} 模型...")
    train_dpo_model(
        data_path=f"./datasets/dpo_converted/{pos_name}_dpo.csv",
        save_path=f"./dpo_outputs/model_{pos}_3B"
    )

    print("🎉 所有模型训练完成。")

if __name__ == "__main__":
    main()
