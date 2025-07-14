import os
import pandas as pd

# === 配置路径（T模型 - SST2）===
base_path = "results/sentiment/sst2/T性格模型"
file_name = "sst2_sentiment_results.csv"
label_suffix = ".invalid.csv"
merge_key = "text"
pred_col = "prediction"

# === 构造完整路径 ===
input_file = os.path.join(base_path, file_name)
label_file = input_file.replace(".csv", label_suffix)
output_file = input_file.replace(".csv", ".processed.csv")

# === 校验文件存在性 ===
if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ 缺失原始文件：{input_file}")
if not os.path.exists(label_file):
    raise FileNotFoundError(f"❌ 缺失标注文件：{label_file}")

# === 读取文件 ===
df_main = pd.read_csv(input_file)
df_label = pd.read_csv(label_file)

# === 检查必要列 ===
if merge_key not in df_main.columns or merge_key not in df_label.columns:
    raise ValueError(f"❌ 缺少合并字段 {merge_key}")
if "sentiment_label" not in df_label.columns:
    raise ValueError("❌ 缺少 sentiment_label 列")

# === 构造替换字典并应用 ===
df_label = df_label.dropna(subset=[merge_key, "sentiment_label"])
label_map = dict(zip(df_label[merge_key].astype(str), df_label["sentiment_label"].astype(str)))

df_main[pred_col] = df_main.apply(
    lambda row: label_map.get(str(row[merge_key]), row[pred_col]),
    axis=1
)

# === 保存结果 ===
df_main.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"✅ SST2 T模型处理完成，结果保存至：{output_file}")
