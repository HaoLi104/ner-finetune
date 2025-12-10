from sentence_transformers import SentenceTransformer, util
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
try:
    from model_path_conf import DEFAULT_SENTENCE_EMBEDDING_PATH  # type: ignore
except Exception:
    DEFAULT_SENTENCE_EMBEDDING_PATH = "/mnt/windows/Users/Admin/LLM/models/embedding"
import torch

# 选择医学句子嵌入模型，例如 coROM
model = SentenceTransformer(DEFAULT_SENTENCE_EMBEDDING_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# 查询文本与候选列表
surgery_query = "双乳切除+双侧胶窝淋巴结标记清扫+双侧筋膜,组织瓣形成术"
surgery_options = [
    "保乳手术", "乳房切除术", "腋窝淋巴结清扫",
    "前哨淋巴活检", "乳房重建", "卵巢切除", "局部病灶姑息性手术"
]

# 批量编码
query_emb = model.encode(surgery_query, convert_to_tensor=True)           # 查询句向量
opts_emb = model.encode(surgery_options, convert_to_tensor=True)         # 候选句向量

# 计算余弦相似度
cos_scores = util.cos_sim(query_emb, opts_emb)[0]                         # 1×N
scores = cos_scores.cpu().tolist()
threshold = 0.5  # 或根据业务需求调优
ranked = sorted(zip(surgery_options, scores), key=lambda x: x[1], reverse=True)

print("候选手术名称相似度排名：")
for opt, score in ranked:
    mark = "✅" if score >= threshold else "❌"
    print(f"{opt:12s}  相似度={score:.4f} {mark}")
