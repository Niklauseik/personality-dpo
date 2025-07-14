import pandas as pd
import os
import re

# === 文件配置 ===
base_dir = "results/medical"
model_dirs = ["原始基座模型", "F性格模型", "T性格模型"]
input_filename = "mental_sentiment_results.processed.csv"

# === 规范化函数 ===
def normalize_prediction(pred):
    if not isinstance(pred, str):
        return pred
    pred_clean = pred.strip().lower()
    if re.match(r"^[a-z]+$", pred_clean):
        return pred_clean
    if "depression" in pred_clean:
        return "depression"
    if "normal" in pred_clean:
        return "normal"
    return pred  # 其他情况保持原样

# === 主处理流程 ===
for model in model_dirs:
    input_path = os.path.join(base_dir, model, input_filename)
    
    if not os.path.exists(input_path):
        print(f"⚠️ 未找到文件：{input_path}")
        continue

    df = pd.read_csv(input_path)
    df["prediction"] = df["prediction"].apply(normalize_prediction)
    df.to_csv(input_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已更新 prediction 字段：{input_path}")
