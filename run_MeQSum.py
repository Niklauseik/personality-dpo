import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 模型路径（base / F / T）===
model_configs = {
    "原始基座模型": "./llama-3B-Instruct",
    "F性格模型": "./dpo_outputs/model_f_3B",
    "T性格模型": "./dpo_outputs/model_t_3B"
}

# === 推理函数 ===
def local_generate(prompt, tokenizer, model):
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# === 测试 MeQSum 全量样本并保存结果 ===
for model_name, model_path in model_configs.items():
    print(f"\n🧪 正在测试模型：{model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    df_meq = pd.read_excel("datasets/medical/MeQSum.xlsx")
    predictions = []

    few_shot_example = """You are a medical assistant. Your task is to help summarise patient messages by extracting the core medical question they are asking.

Background:
Patients often write informal or verbose messages describing symptoms, personal concerns, or medication experiences. Your job is to identify their underlying medical question and rewrite it clearly and concisely as a single English question. 
Do not add information or rephrase with assumptions. Only reword what is explicitly asked.

Your output must always be a single, direct question. If no question is clearly stated, infer it faithfully based on the message’s intent.

---

Example:
Input: I need/want to know who manufactures Cetirizine. My Walmart is looking for a new supply and are not getting the recent.  
Output: Who manufactures cetirizine?
"""

    for idx, row in tqdm(df_meq.iterrows(), total=len(df_meq), desc=f"→ {model_name}"):
        chq = str(row["CHQ"]).strip()
        prompt = few_shot_example + f"\n\n---\n\nNow process the following:\nInput: {chq}\nOutput:"
        try:
            prediction = local_generate(prompt, tokenizer, model)
        except Exception as e:
            prediction = f"[Error] {e}"

        predictions.append(prediction)

        print("\n-----------------------------")
        print(f"📌 模型：{model_name}")
        print(f"📎 Index: {idx}")
        print(f"📝 CHQ: {chq}")
        print(f"📤 Prediction: {prediction}")

    # === 保存 CSV 文件 ===
    df_result = df_meq.copy()
    df_result["prediction"] = predictions
    save_dir = os.path.join("results", "medical", model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "meqsum_results.csv")
    df_result.to_csv(save_path, index=False)
    print(f"✅ 保存完成：{save_path}")
