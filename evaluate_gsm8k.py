import os
import pandas as pd
import re

# === 模型名称与结果路径 ===
model_names = ["F性格模型", "T性格模型", "原始基座模型"]
base_dir = "./results/benchmark"
result_file = "gsm8k_test800_results.csv"
output_txt = os.path.join(base_dir, "gsm8k_metrics_summary.txt")

# === 数字提取函数 ===
def extract_numbers(text):
    text = str(text).replace(",", "").replace("$", "")
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]

# === 评估主逻辑 ===
summary_lines = []

for model in model_names:
    path = os.path.join(base_dir, model, result_file)
    if not os.path.exists(path):
        summary_lines.append(f"❌ 模型 {model} 缺少文件：{result_file}\n")
        continue

    df = pd.read_csv(path)
    correct, total = 0, 0

    for _, row in df.iterrows():
        label_nums = extract_numbers(row["label"])
        pred_nums = extract_numbers(row["prediction"])
        if not label_nums or not pred_nums:
            continue
        label = label_nums[0]
        if label in pred_nums:
            correct += 1
        total += 1

    acc = correct / total if total else 0
    summary_lines.append(
        f"📌 模型：{model}\n"
        f"✅ Accuracy: {acc:.4f}（{correct}/{total}）\n"
        + "-" * 40 + "\n"
    )

# === 保存结果 ===
with open(output_txt, "w", encoding="utf-8") as f:
    f.writelines(summary_lines)

print("✅ GSM8K 评估完成，结果已保存至：", output_txt)
