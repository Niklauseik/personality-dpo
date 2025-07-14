import pandas as pd
import re

input_file = "results/finbench/F性格模型/fiqasa_results.processed.csv"
output_file = "results/finbench/F性格模型/fiqasa_results.fixed.csv"

def split_line(line: str):
    line = line.strip()
    # 提取最后两个英文词作为标签
    match = re.match(r"^(.*?)(\s+)(\w+)\s+(\w+)$", line)
    if not match:
        return None
    text = match.group(1).strip()
    answer = match.group(3).strip()
    prediction = match.group(4).strip()
    return {
        "text": text,
        "answer": answer,
        "prediction": prediction
    }

with open(input_file, "r", encoding="utf-8-sig") as f:
    lines = [line for line in f if line.strip()]
    rows = [split_line(line) for line in lines[1:]]  # 跳过表头
    rows = [r for r in rows if r]

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"✅ 已修复并保存：{output_file}")
