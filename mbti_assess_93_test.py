import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ 三个模型配置（路径根据你已有目录）
model_configs = {
    "原始基座模型": "./llama-3B-Instruct",
    "F性格模型": "./dpo_outputs/model_f_3B",
    "T性格模型": "./dpo_outputs/model_t_3B"
}

# ✅ 固定一条样本，生成两个顺序版本
test_prompts = [
    {
        "label": "F→T 顺序",
        "question": "Do you often make decisions in a way that...",
        "choice_1": "Your emotions dominate your intellect",  # F
        "choice_2": "Your intellect dominates your emotions"   # T
    },
    {
        "label": "T→F 顺序",
        "question": "Do you often make decisions in a way that...",
        "choice_1": "Your intellect dominates your emotions",   # T
        "choice_2": "Your emotions dominate your intellect"     # F
    }
]

# ✅ 推理参数
gen_kwargs = dict(
    max_new_tokens=128,
    do_sample=True,
    temperature=0.1,
    top_p=1,
    repetition_penalty=1.1
)

def run_test(model_name, model_path):
    print(f"\n🧠 模型：{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    for idx, item in enumerate(test_prompts):
        question = item["question"]
        option1 = item["choice_1"]
        option2 = item["choice_2"]
        label = item["label"]

        # ✅ Prompt 构造
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": (
                "Pretend you are a real human being. "
                "Answer as if you were truly a person with preferences.\n\n"
                f"{question}\n\n"
                f"1. {option1}\n"
                f"2. {option2}\n\n"
                "Which one fits you better? Reply with the full sentence."
            )}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # ✅ 去除多余头部，仅保留模型回答
        if "assistant" in full_output.lower():
            response = full_output.lower().split("assistant", 1)[1].strip()
        else:
            response = full_output

        print(f"[{model_name}] ({label}) Q: {question} | 🤖 回答: {response}")

if __name__ == "__main__":
    for name, path in model_configs.items():
        run_test(model_name=name, model_path=path)
