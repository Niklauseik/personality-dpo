import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 模型路径配置 ===
model_configs = {
    "原始基座模型": "./llama-3B-Instruct",
    "F性格模型": "./dpo_outputs/model_f_3B",
    "T性格模型": "./dpo_outputs/model_t_3B"
}

# === 加载数据集（保持原字段）===
movie_data = pd.read_csv("datasets/movie/wiki_movie_summ_3k.csv")   # 列：Plot, PlotSummary
imdb_data = pd.read_csv("datasets/movie/imdb_test.csv")             # 列：text, label

# === 推理函数 ===
def local_generate(prompt, tokenizer, model):
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# === 主流程：所有模型 × 2个任务 ===
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

    # === Movie 摘要生成 ===
    movie_predictions = []
    for _, row in tqdm(movie_data.iterrows(), total=len(movie_data), desc=f"{model_name} - Movie"):
        prompt = (
            "You are a helpful assistant that writes short summaries for movie plots.\n\n"
            f"Plot:\n{row['Plot']}\n\nSummary:"
        )
        try:
            pred = local_generate(prompt, tokenizer, model)
        except Exception as e:
            pred = f"[Error] {e}"
        movie_predictions.append(pred)

    movie_result = movie_data.copy()
    movie_result["prediction"] = movie_predictions
    movie_out_dir = os.path.join("results", "movie_summary", model_name)
    os.makedirs(movie_out_dir, exist_ok=True)
    movie_result.to_csv(os.path.join(movie_out_dir, "movie_summary_results.csv"), index=False, encoding="utf-8")
    print(f"✅ Movie 摘要结果已保存：{movie_out_dir}")

    # === IMDb 情感分类 ===
    imdb_predictions = []
    for _, row in tqdm(imdb_data.iterrows(), total=len(imdb_data), desc=f"{model_name} - IMDb"):
        prompt = (
            "You are a movie review sentiment classifier. Respond with only one word: positive or negative.\n\n"
            f"Review:\n{row['text']}\n\nSentiment:"
        )
        try:
            pred = local_generate(prompt, tokenizer, model)
        except Exception as e:
            pred = f"[Error] {e}"
        imdb_predictions.append(pred)

    imdb_result = imdb_data.copy()
    imdb_result["prediction"] = imdb_predictions
    imdb_out_dir = os.path.join("results", "imdb_sentiment", model_name)
    os.makedirs(imdb_out_dir, exist_ok=True)
    imdb_result.to_csv(os.path.join(imdb_out_dir, "imdb_sentiment_results.csv"), index=False, encoding="utf-8")
    print(f"✅ IMDb 结果已保存：{imdb_out_dir}")
