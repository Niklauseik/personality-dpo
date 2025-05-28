import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 三个模型路径（base / T / F）
model_paths = {
    "F性格模型": "./dpo_outputs/model_f_3B",
    "T性格模型": "./dpo_outputs/model_t_3B",
    "原始基座模型": "./llama-3B-Instruct"
}

# ✅ 加载数据集的前 3 条
datasets = {
    "gsm8k_test800": pd.read_csv("datasets/benchmark/gsm8k_test800.csv").head(3),
    "arc_easy_test800": pd.read_csv("datasets/benchmark/arc_easy_test800.csv").head(3),
    "boolq_train800": pd.read_csv("datasets/benchmark/boolq_train800.csv").head(3)
}

# ✅ 推理函数（可复用）
def local_generate(prompt, tokenizer, model):
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ✅ 遍历模型 × 数据集
for model_name, model_path in model_paths.items():
    print(f"\n==================== 🧠 正在测试模型：{model_name} ====================")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    for dataset_name, df in datasets.items():
        print(f"\n📊 数据集：{dataset_name}\n")

        for i, row in df.iterrows():
            # === 构造 user prompt ===
            if "gsm8k" in dataset_name:
                prompt = (
                    f"Solve the following math problem and output only the final number answer.\n\n"
                    f"{row['question']}\n\nOnly respond with one word. like 8"
                )
            elif "arc_easy" in dataset_name:
                prompt = (
                    f"Choose the correct option (A/B/C/D) for the following question.\n\n"
                    f"Question: {row['question']}\nOptions:\n{row['choices']}\n\n"
                    f"Only respond with one word. like: A"
                )
            elif "boolq" in dataset_name:
                prompt = (
                    f"Based on the passage, answer whether the question is true or false.\n\n"
                    f"Passage: {row['passage']}\n\nQuestion: {row['question']}\n\n"
                    f"Only respond with one word. like: true"
                )
            else:
                prompt = row["question"]

            try:
                answer = local_generate(prompt, tokenizer, model)
            except Exception as e:
                answer = f"[Error] {e}"

            print(f"[{dataset_name} Q{i+1}] → {answer}")
