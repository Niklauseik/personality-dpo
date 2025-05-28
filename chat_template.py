from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 三个模型路径（base / T / F）
model_paths = {
    "F性格模型": "./dpo_outputs/model_f_3B",
    "T性格模型": "./dpo_outputs/model_t_3B",
    "原始基座模型": "./llama-3B-Instruct"
}

# ✅ 问题
messages = [
    {"role": "user", "content": "假如你考试失利了，你会怎么做？"}
]

# ✅ 推理参数
gen_kwargs = dict(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.2,
    top_p=0.8,
    repetition_penalty=1.1
)

# ✅ 遍历模型测试
for name, model_path in model_paths.items():
    print(f"\n=== 🔍 正在测试模型：{name} ===")

    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    # 构造 prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)

    # 生成
    with torch.no_grad():
        out = model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            **gen_kwargs
        )

    # 解码新生成部分
    generated = out[0][ inputs["input_ids"].shape[-1]: ]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(f"🗨️ 回答：{answer}")
