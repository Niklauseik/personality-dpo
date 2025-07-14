import re

input_path = "results/sentiment/sst2/T性格模型/sst2_sentiment_results.processed.csv"

# 读取错误格式的内容
with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 修复内容
fixed_lines = []
fixed_lines.append("idx,text,label,prediction\n")  # 添加正确表头

for line in lines[1:]:  # 跳过第一行错误表头
    line = line.strip()
    if not line:
        continue

    # 使用正则提取 idx、text、label、prediction
    match = re.match(r'^"?(\d+)[\s,]*(.*?)[\s,]+"?([01])"?[\s,]+"?(positive|negative|neutral|Positive|Negative|Neutral)"?$', line)
    if not match:
        print(f"⚠️ 跳过格式异常行：{line}")
        continue

    idx, text, label, pred = match.groups()
    # 转义双引号
    text = text.replace('"', '""')
    fixed_lines.append(f'{idx},"{text}",{label},{pred.lower()}\n')

# 原地写回原文件
with open(input_path, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(fixed_lines)

print("✅ 修复完成：sst2_sentiment_results.processed.csv")
