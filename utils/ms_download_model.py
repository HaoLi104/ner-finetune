from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-Embedding-0.6B",
    cache_dir="/mnt/windows/Users/Admin/LLM/models/Qwen3-Embedding-0.6B",
    ignore_file_pattern=[
        "tf_model.h5",
    ],
)
