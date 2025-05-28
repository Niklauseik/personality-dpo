import os
import time
import torch
import pandas as pd
import collections
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ 三个模型配置
model_configs = {
    "原始基座模型": "./llama-3B-Instruct",
    "F性格模型": "./dpo_outputs/model_f_3B_previous",
    "T性格模型": "./dpo_outputs/model_t_3B_previous"
}

# ✅ 数据路径（新版）
MBTI_DATASET = "datasets/MBTI_doubled_93.json"
RESULTS_ROOT = "results/mbti_types"
NUM_TRIALS = 10
EARLY_STOP_COUNT = 6

# ✅ 推理参数
gen_kwargs = dict(
    max_new_tokens=128,
    do_sample=True,
    temperature=0.1,
    top_p=1,
    repetition_penalty=1.1
)

# ✅ 加载数据集
df = pd.read_json(MBTI_DATASET)

# ✅ 主测试函数
def run_mbti_test(model_name, model_path):
    print(f"\n🧠 正在测试模型：{model_name}")
    save_dir = os.path.join(RESULTS_ROOT, model_name)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    mbti_count = collections.Counter()
    dimension_counts = {dim: collections.Counter() for dim in ["E/I", "S/N", "T/F", "J/P"]}

    for trial in range(NUM_TRIALS):
        print(f"🔄 第 {trial+1}/{NUM_TRIALS} 次测试")

        dimension_count = collections.Counter({k: 0 for k in "EISNTFJP"})
        valid_predictions = 0

        for idx, row in df.iterrows():
            question = row["question"]
            a_text = row["choice_a"]["text"]
            b_text = row["choice_b"]["text"]
            a_value = row["choice_a"]["value"]
            b_value = row["choice_b"]["value"]

            user_prompt = (
                "You are answering an MBTI personality test.\n"
                "For the question, select either choice_a or choice_b.\n"
                "Respond with only one word: 'a' or 'b'.\n\n"
                f"Q{idx + 1}: {question}\n"
                f"A: {a_text} (a)\n"
                f"B: {b_text} (b)"
            )

            messages = [{"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            ans = decoded[-1] if decoded else ""

            if ans not in {"a", "b"}:
                continue

            valid_predictions += 1
            chosen_value = a_value if ans == "a" else b_value
            dimension_count[chosen_value] += 1

        if valid_predictions == 0:
            print("⚠️ 无有效回答，跳过该轮")
            continue

        mbti_result = "".join([
            "E" if dimension_count["E"] >= dimension_count["I"] else "I",
            "S" if dimension_count["S"] >= dimension_count["N"] else "N",
            "T" if dimension_count["T"] >= dimension_count["F"] else "F",
            "J" if dimension_count["J"] >= dimension_count["P"] else "P"
        ])

        mbti_count[mbti_result] += 1
        dimension_counts["E/I"][mbti_result[0]] += 1
        dimension_counts["S/N"][mbti_result[1]] += 1
        dimension_counts["T/F"][mbti_result[2]] += 1
        dimension_counts["J/P"][mbti_result[3]] += 1

        most_common = mbti_count.most_common(1)
        if most_common and most_common[0][1] >= EARLY_STOP_COUNT:
            print(f"⏹️ 类型 {most_common[0][0]} 出现 {most_common[0][1]} 次，提前终止")
            break

        time.sleep(0.2)

    # 汇总结果
    most_common_mbti = mbti_count.most_common(1)[0][0] if mbti_count else "N/A"
    most_common_by_dim = "".join([
        max(dimension_counts["E/I"], key=dimension_counts["E/I"].get, default="?"),
        max(dimension_counts["S/N"], key=dimension_counts["S/N"].get, default="?"),
        max(dimension_counts["T/F"], key=dimension_counts["T/F"].get, default="?"),
        max(dimension_counts["J/P"], key=dimension_counts["J/P"].get, default="?")
    ])

    final_report = f"""
✅ MBTI 多次测试完成（最多 {NUM_TRIALS} 次，实际运行 {sum(mbti_count.values())} 次）

📌 完整 MBTI 类型统计
{mbti_count}

📌 按维度统计
E/I: {dimension_counts["E/I"]}
S/N: {dimension_counts["S/N"]}
T/F: {dimension_counts["T/F"]}
J/P: {dimension_counts["J/P"]}

📊 方式一：完整类型最多 → {most_common_mbti}
📊 方式二：各维度最多 → {most_common_by_dim}
"""
    result_file = os.path.join(save_dir, "final_mbti_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(final_report)
    print(f"📁 结果已保存至：{result_file}")


if __name__ == "__main__":
    for name, path in model_configs.items():
        run_mbti_test(model_name=name, model_path=path)
