from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    local_dir="./llama-3B-Instruct",
    local_dir_use_symlinks=False  # 可避免软链接导致复制不全
)
