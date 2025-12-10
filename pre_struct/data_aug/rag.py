# rag.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import re
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm

# ---------------- logger ----------------
log = logging.getLogger("rag")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    )
    log.addHandler(h)
    log.setLevel(logging.INFO)

# ---------------- TokenCounter（保留兼容） ----------------
try:
    from transformers import BertTokenizerFast  # type: ignore
except Exception:
    BertTokenizerFast = None  # type: ignore


try:
    from model_path_conf import DEFAULT_TOKENIZER_PATH, DEFAULT_EMBEDDING_MODEL_PATH  # type: ignore
except Exception as exc:
    raise ImportError(
        "DEFAULT_TOKENIZER_PATH and DEFAULT_EMBEDDING_MODEL_PATH must be defined in model_path_conf.py"
    ) from exc


class TokenCounter:
    def __init__(
        self,
        tokenizer_name: str = DEFAULT_TOKENIZER_PATH,
    ) -> None:
        self.tokenizer = None
        if BertTokenizerFast is not None:
            try:
                self.tokenizer = BertTokenizerFast.from_pretrained(
                    tokenizer_name, local_files_only=True, trust_remote_code=True
                )
            except Exception:
                self.tokenizer = None
        self._basic_re = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+|[^\s\w]", re.UNICODE)

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.tokenize(text))
            except Exception:
                pass
        return len(self._basic_re.findall(text or ""))


# LangChain-only implementation

# ---------------- 可选依赖：LangChain HuggingFaceEmbeddings ----------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings as LC_HFEmb  # type: ignore

    _HAS_LC_HF = True
except Exception:
    LC_HFEmb = None  # type: ignore
    _HAS_LC_HF = False

# ---------------- LangChain 文本加载与切块 ----------------
try:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    from langchain_community.document_loaders import TextLoader  # type: ignore

    _HAS_LC_CORE = True
except Exception:
    RecursiveCharacterTextSplitter = None  # type: ignore
    TextLoader = None  # type: ignore
    _HAS_LC_CORE = False


# ---------------- Embedders Base ----------------
class _BaseEmbedder:
    def encode(
        self, texts: List[str], batch_size: int = 32, progress: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


    # LangChain embedder only


# ---------------- LangChain HuggingFace Embedder ----------------
class _LangChainHFEmbedder(_BaseEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        normalize: bool = False,
    ):
        if not _HAS_LC_HF:
            raise RuntimeError("langchain-huggingface 未安装")
        model_kwargs = {}
        if device:
            model_kwargs["device"] = device
        self._emb = LC_HFEmb(
            model_name=model_name_or_path,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": normalize},
        )
        vec = self._emb.embed_query("test")
        self._dim = len(vec)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(
        self, texts: List[str], batch_size: int = 32, progress: bool = False
    ) -> np.ndarray:
        bs = max(1, int(batch_size))
        out: List[List[float]] = []
        rng = range(0, len(texts), bs)
        iterator = (
            tqdm(rng, desc="RAG encode (LangChain HF)", ncols=90) if progress else rng
        )
        for i in iterator:
            chunk = texts[i : i + bs]
            out.extend(self._emb.embed_documents(chunk))
        return (
            np.asarray(out, dtype="float32")
            if out
            else np.zeros((0, self.dim), dtype="float32")
        )


# ---------------- 构造 Embedder（保持函数名/签名不变） ----------------
def _build_embedder(embedding_model: str, use_fp16: Optional[bool]) -> _BaseEmbedder:
    """Build LangChain HuggingFaceEmbeddings (supports 'lc:' prefix)."""
    _ = use_fp16  # 接口兼容参数，不再使用
    mp = str(embedding_model or "").strip()
    if mp.startswith("lc:"):
        mp = mp[3:]
    if not _HAS_LC_HF:
        raise RuntimeError("langchain-huggingface 未安装，无法构建 Embedder")
    device = os.getenv("RAG_LC_DEVICE") or None
    log.info(f"[RAG] use LangChain HuggingFaceEmbeddings: {mp} (device={device})")
    return _LangChainHFEmbedder(mp, device=device, normalize=False)


def _make_chunks_from_file(path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Load file and split via LangChain (TextLoader + RecursiveCharacterTextSplitter)."""
    if not _HAS_LC_CORE:
        raise RuntimeError("需要安装 langchain 文本加载与切块依赖 (langchain_community, langchain_text_splitters)")
    loader = TextLoader(path, autodetect_encoding=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(50, int(chunk_size)),
        chunk_overlap=max(0, min(int(chunk_overlap), int(chunk_size) // 2)),
        length_function=len,
        add_start_index=False,
        separators=["。", "\n\n", "\n"]
        if hasattr(RecursiveCharacterTextSplitter, "from_tiktoken_encoder") is False
        else None,
    )
    if hasattr(splitter, "split_documents"):
        pieces = splitter.split_documents(docs)
        chunks = [d.page_content.strip() for d in pieces]
    else:
        chunks = []
        for d in docs:
            chunks.extend([t.strip() for t in splitter.split_text(d.page_content) if t.strip()])
    return [c for c in chunks if c]


# ---------------- RAG Index ----------------
class RAGIndex:
    """Cosine similarity search; cached under data/.rag_cache."""

    def __init__(self, chunks: List[str], emb: np.ndarray):
        self.chunks = chunks
        self._emb = emb.astype("float32")
        norms = np.linalg.norm(self._emb, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        self._emb_n = self._emb / norms

    def ready(self) -> bool:
        return bool(self.chunks) and self._emb_n.size > 0

    def search(
        self, query: str, topk: int = 3, embedder: Optional[_BaseEmbedder] = None
    ) -> List[Tuple[int, float]]:
        if not self.ready() or not query:
            return []
        if embedder is None:
            raise RuntimeError("RAGIndex.search 需要传入 embedder（内部缓存时会设置）")
        q = embedder.encode([query], batch_size=1, progress=False)  # (1, d)
        qn = q / max(1e-6, float(np.linalg.norm(q)))
        sims = (self._emb_n @ qn.reshape(-1, 1)).reshape(-1)  # cosine
        idx = np.argsort(-sims)[: max(1, int(topk))]
        return [(int(i), float(sims[i])) for i in idx]


# ---------------- 缓存：读写（保留） ----------------
def _cache_dir() -> Path:
    p = Path("data/.rag_cache")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash_cfg(
    rag_path: str, chunk_size: int, chunk_overlap: int, model_id: str
) -> str:
    key = json.dumps(
        {
            "ver": 6,
            "p": Path(rag_path).resolve().as_posix(),
            "cs": int(chunk_size),
            "co": int(chunk_overlap),
            "m": str(model_id),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _cache_paths(key: str) -> Dict[str, Path]:
    base = _cache_dir() / f"rag_{key}"
    return {
        "meta": base.with_suffix(".meta.json"),
        "emb": base.with_suffix(".emb.npy"),
        "chunks": base.with_suffix(".chunks.jsonl"),
    }


def _save_cache(
    key: str, chunks: List[str], emb: np.ndarray, meta: Dict[str, Any]
) -> None:
    paths = _cache_paths(key)
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    tmp_chunks = paths["chunks"].with_suffix(paths["chunks"].suffix + ".tmp")
    tmp_emb = paths["emb"].with_suffix(paths["emb"].suffix + ".tmp")
    tmp_meta = paths["meta"].with_suffix(paths["meta"].suffix + ".tmp")

    with tmp_chunks.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
    with tmp_emb.open("wb") as f:
        np.save(f, emb.astype("float32"))
    tmp_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        tmp_chunks.replace(paths["chunks"])
        tmp_meta.replace(paths["meta"])
        tmp_emb.replace(paths["emb"])
    except Exception as e:
        log.warning(f"[RAG] 写缓存失败：{e}")


def _load_cache(
    key: str,
) -> Tuple[Optional[List[str]], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    paths = _cache_paths(key)
    if not (
        paths["meta"].exists() and paths["emb"].exists() and paths["chunks"].exists()
    ):
        return None, None, None
    try:
        chunks: List[str] = []
        with paths["chunks"].open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        chunks.append(obj["text"])
                except Exception:
                    continue
        with paths["emb"].open("rb") as f:
            emb = np.load(f)
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        return chunks, emb, meta
    except Exception as e:
        log.warning(f"[RAG] 读取缓存失败，将重建：{e}")
        return None, None, None


# ---------------- 入口：构建/读取缓存（函数名/签名保持） ----------------
class _CachedRAG:
    def __init__(self, index: RAGIndex, embedder: _BaseEmbedder):
        self._index = index
        self._embedder = embedder

    def ready(self) -> bool:
        return self._index.ready()

    @property
    def chunks(self) -> List[str]:
        return self._index.chunks

    def search(self, query: str, topk: int = 3) -> List[Tuple[int, float]]:
        return self._index.search(query, topk=topk, embedder=self._embedder)


def get_cached_rag_index(
    rag_path: str,
    chunk_size: int = 100,
    chunk_overlap: int = 10,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_batch_size: int = 64,
    show_progress: bool = True,
    use_fp16: Optional[bool] = None,
) -> _CachedRAG:
    """Build/load cached RAG index using LangChain backends."""
    if not rag_path or not Path(rag_path).exists():
        raise FileNotFoundError(f"RAG 语料不存在：{rag_path}")

    log.info(
        f"[RAG] request: path={rag_path} | model={embedding_model} | chunk={chunk_size}/{chunk_overlap}"
    )
    # 仅使用 LangChain 作为切块后端
    if not _HAS_LC_CORE:
        raise RuntimeError("需要安装 langchain 文本加载与切块依赖 (langchain_community, langchain_text_splitters)")
    key = _hash_cfg(rag_path, chunk_size, chunk_overlap, embedding_model)
    paths = _cache_paths(key)
    log.info(
        f"[RAG] cache key={key} | cache files: {paths['meta'].name}, {paths['emb'].name}, {paths['chunks'].name}"
    )

    chunks, emb, meta = _load_cache(key)
    if chunks is not None and emb is not None:
        log.info(
            f"[RAG] 命中缓存（key={key}，chunks={len(chunks)}，dim={emb.shape[-1]}）"
        )
        embedder = _build_embedder(embedding_model, use_fp16=use_fp16)
        return _CachedRAG(RAGIndex(chunks, emb), embedder)

    # 构建
    t0 = time.time()
    log.info("[RAG] 构建索引（切块中，chunk_backend=langchain)...")
    chunks = _make_chunks_from_file(rag_path, chunk_size, chunk_overlap)
    log.info(f"[RAG] 切块完成：{len(chunks)} 段")

    embedder = _build_embedder(embedding_model, use_fp16=use_fp16)
    log.info(f"[RAG] embedder dim={embedder.dim}，开始编码 ...")
    emb = embedder.encode(chunks, batch_size=encode_batch_size, progress=show_progress)
    if emb.shape[0] != len(chunks):
        raise RuntimeError(
            f"向量数与块数不一致：emb={emb.shape} vs chunks={len(chunks)}"
        )

    index = RAGIndex(chunks, emb)
    meta = {
        "rag_path": str(Path(rag_path).resolve()),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "embedding_model": str(embedding_model),
        "dim": int(embedder.dim),
        "num_chunks": len(chunks),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "took_s": round(time.time() - t0, 3),
        "ver": 6,
        "batch": int(encode_batch_size),
        "backend": "langchain",
    }
    _save_cache(key, chunks, emb, meta)
    log.info(f"[RAG] 缓存已写入 data/.rag_cache（key={key}，chunks={len(chunks)}）")
    return _CachedRAG(index, embedder)


# ---------------- Quick test ----------------
def quick_test_search(
    rag_path: str,
    query: str,
    *,
    topk: int = 3,
    chunk_size: int = 100,
    chunk_overlap: int = 10,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL_PATH,
    encode_batch_size: int = 128,
    show_progress: bool = True,
    use_fp16: Optional[bool] = True,
    max_chars: int = 300,
) -> List[Dict[str, Any]]:
    """Build/load index and return top-k matches as previews."""

    rag = get_cached_rag_index(
        rag_path=rag_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        encode_batch_size=encode_batch_size,
        show_progress=show_progress,
        use_fp16=use_fp16,
    )
    if not rag.ready():
        raise RuntimeError("RAG 索引未就绪")

    hits = rag.search(query, topk=topk)
    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(hits, 1):
        txt = rag.chunks[idx]
        cut = (txt[:max_chars] + "…") if len(txt) > max_chars else txt
        results.append(
            {
                "rank": rank,
                "score": round(float(score), 6),
                "index": int(idx),
                "text": cut,
            }
        )
    return results


# ---------------- 示例 main：无 argparse，直接函数传参 ----------------
if __name__ == "__main__":
    DEMO_RAG_PATH = "data/report_chunks.txt"
    DEMO_QUERY = ""
    DEMO_PARAMS = dict(
        topk=3,
        chunk_size=300,
        chunk_overlap=30,
        embedding_model="/mnt/windows/Users/Admin/LLM/models/Qwen3-Embedding-0.6B/Qwen/Qwen3-Embedding-0.6B/",
        encode_batch_size=64,
        show_progress=True,
        use_fp16=True,
        max_chars=300,
    )

    rows = quick_test_search(
        rag_path=DEMO_RAG_PATH,
        query=DEMO_QUERY,
        **DEMO_PARAMS,
    )

    print("\n=== Top-{} Matches ===".format(DEMO_PARAMS["topk"]))
    for r in rows:
        print(f"[#{r['rank']}] score={r['score']:.6f} | idx={r['index']}")
        print(r["text"])
        print("-" * 60)
