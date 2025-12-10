"""
Centralized model/tokenizer paths for easy maintenance.

Override these values as needed in your environment.
Other modules can import this module instead of hardcoding paths.
"""

# Default pretrained model directory (local path or HuggingFace id)
DEFAULT_MODEL_PATH = "/mnt/windows/Users/Admin/LLM/models/bert_multi_language/"
# DEFAULT_MODEL_PATH = "/data/ocean/model/google-bert/bert-base-multilingual-cased/"

# Default tokenizer path; by default reuse model path
DEFAULT_TOKENIZER_PATH = "/mnt/windows/Users/Admin/LLM/models/bert_multi_language/"
# DEFAULT_TOKENIZER_PATH = "/data/ocean/model/google-bert/bert-base-multilingual-cased/"

# Common embedding model for RAG (Qwen3-Embedding)
DEFAULT_EMBEDDING_MODEL_PATH = (
    "/mnt/windows/Users/Admin/LLM/models/Qwen3-Embedding-0.6B/"
)

# Long context model directory (if used by tools)
DEFAULT_LONGFORMER_PATH = "/mnt/windows/Users/Admin/LLM/models/longformer"

# SentenceTransformer local directory (if used by utils/test_similarity.py)
DEFAULT_SENTENCE_EMBEDDING_PATH = "/mnt/windows/Users/Admin/LLM/models/embedding"
