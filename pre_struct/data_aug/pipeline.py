# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import time
import copy
import random
import threading
import concurrent.futures
import hashlib
from difflib import SequenceMatcher
from queue import Queue, Empty
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    from model_path_conf import DEFAULT_TOKENIZER_PATH  # type: ignore
except Exception as exc:
    raise ImportError(
        "DEFAULT_TOKENIZER_PATH must be defined in model_path_conf.py"
    ) from exc

from tqdm import tqdm

from llm_client import OpenAIFieldWiseLLM
from augmenter import LLMPromptAugmenter
from structs import REPORT_STRUCTURE_MAP  # dict[str, list[str]]

# ----------------------------- 常量/工具 -----------------------------
RESERVED_KEYS = {"report", "report_title", "report_titles", "meta"}

# 合成排除：不适合由 LLM 生成的“管理/编号/时间类”字段（容易产生不一致或敏感信息）
SYNTH_EXCLUDE_KEYS = {
    "报告类型", "报告时间", "报告日期", "检查时间", "检查日期",
    "检查号", "登记号", "住院号", "门诊号", "病员号", "床号", "超声号",
    "联系电话", "联系方式", "录入员", "申请医生", "检查医生", "审核医生", "医院名",
}


def _normalize_title_for_counting(x: str) -> str:
    x = str(x or "")
    x = x.replace("：", ":").replace("（", "(").replace("）", ")")
    x = re.sub(r"\s+", "", x)
    return x.strip()


def _norm_text_for_hash(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = s.replace("\r", "\n")
    s = re.sub(r"\s+", "", s).lower()
    return s


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ----------------------------- 异步 JSONL Writer -----------------------------
class _AsyncSaver:
    """把增强结果持续写到 JSONL（out_path + '.aug.jsonl'）。"""

    def __init__(self, checkpoint_path: Path, queue_maxsize: int = 4096) -> None:
        self.checkpoint_path = checkpoint_path
        self.q: "Queue[List[Dict[str, Any]]]" = Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        self._th.join(timeout=10)

    def enqueue(self, items: List[Dict[str, Any]]) -> None:
        self.q.put(items, timeout=60)

    def _loop(self) -> None:
        buf: List[Dict[str, Any]] = []
        last_flush = time.time()
        with self.checkpoint_path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set():
                try:
                    batch = self.q.get(timeout=0.2)
                except Empty:
                    batch = None
                if batch:
                    buf.extend(batch)
                    self.q.task_done()

                now = time.time()
                if buf and (len(buf) >= 20 or (now - last_flush) >= 0.5):
                    for rec in buf:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    buf.clear()
                    last_flush = now

            if buf:
                for rec in buf:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()


# ----------------------------- 主 Pipeline -----------------------------
class DataAugmentPipeline:
    def __init__(
        self,
        # Ⅰ. 路径
        in_path: str = "data/clean_ocr_ppt.json",
        out_path: str = "data/clean_ocr_ppt_da.json",
        # Ⅱ. 新增：为“相对结构映射新增”的 key 生成值，并记录到 added_keys（不改旧值/不改 report）
        inc_synthesize_new_keys: bool = False,  # <<< 开启后：每条记录补出新增 key 的值，并在该记录里加 added_keys 列表
        # Ⅱ+. 组合模式：一次性完成“增量新键合成 + 为缺失标题合成样本至目标均衡值”
        inc_and_synthesize_missing_to_median: bool = False,
        # Ⅲ. 少数类/均衡（默认关闭）
        topk_to_median: bool = False,
        topk_titles: int = 10,
        # Ⅲ+. 统计口径：补齐目标可选 'median' | 'mean'
        fill_stat: str = "mean",
        # Ⅳ. 旧增强模式（默认关闭）
        k_per_record: int = 0,
        augment_minority_only: bool = False,
        minority_threshold: Optional[int] = None,
        k_per_record_minor: Optional[int] = None,
        # Ⅴ. LLM / tokenizer / RAG
        tokenizer_name: str = DEFAULT_TOKENIZER_PATH,
        use_openai: bool = True,
        openai_base: Union[str, List[str]] = "https://qwen3.yoo.la/v1/",
        openai_model: str = "qwen3-32b",
        api_key: Optional[str] = None,
        timeout: int = 20,
        rag_path: Optional[str] = None,
        rag_topk: int = 3,
        include_context_summary: bool = True,
        # Ⅵ. 并发/限流
        reports_workers: int = 4,
        fields_workers: Optional[int] = 8,
        per_base_pool_maxsize: int = 128,
        per_base_max_inflight: int = 24,
        # Ⅶ. 组合风格（默认较保守）
        compose_sep_prob: float = 0.5,
        compose_ocr_noise: bool = False,
        compose_paragraph_prob: float = 0.0,
        value_linebreak_prob: float = 0.0,
        seed: int = 42,
        # Ⅷ. 结构扰动/整条合成（默认关闭）
        exhaustive_titles: bool = False,
        exhaustive_modules: bool = False,
        synthesize_missing_titles: bool = False,
        new_per_title: int = 0,
        synth_keys_drop_prob: float = 0.25,
        synth_min_keys: int = 4,
        synth_max_keys: Optional[int] = None,
        synth_include_basic_info: bool = False,
        struct_perturb_enable: bool = False,
        # Ⅸ. 新增：报告类型过滤和数量控制
        target_report_types: Optional[List[str]] = None,  # 指定要增强的报告类型，None表示所有
        max_samples_per_type: Optional[int] = None,       # 每种报告类型的最大样本数
        max_total_samples: Optional[int] = None,          # 总输出样本数上限
        struct_variants_per_record: int = 0,
        struct_add_prob: float = 0.35,
        struct_drop_prob: float = 0.25,
        struct_min_keys: int = 3,
        # Ⅸ. 写盘/去重（最终合并）
        dedup_during_generation: bool = False,
        final_dedup_mode: str = "exact",
        final_sim_threshold: float = 0.70,
        final_sim_recent_k: int = 500,
        # Ⅹ. 批量
        flush_every: int = 200,
        # ⅩⅠ. 合成防护（避免一次请求字段过多导致 JSON 解析失败）
        synth_batch_size: int = 18,
        # ⅩⅡ. 增量补齐键数量控制（可选）
        inc_max_keys_per_record: Optional[int] = None,
        inc_key_pick: str = "head",  # 'head' | 'random'
        # ⅩⅢ. 字段合成模式：'batched' | 'per_key'
        fields_synth_mode: str = "batched",
    ) -> None:
        if not use_openai:
            raise ValueError("严格模式：use_openai 必须为 True")

        # 路径
        self.in_path = in_path
        self.out_path = out_path

        # 增量 key 合成
        self.inc_synthesize_new_keys = bool(inc_synthesize_new_keys)
        self.inc_and_synthesize_missing_to_median = bool(
            inc_and_synthesize_missing_to_median
        )

        # 少数类/旧模式参数
        self.topk_to_median = bool(topk_to_median)
        self.topk_titles = max(1, int(topk_titles))
        self.k_per_record = max(0, int(k_per_record))
        self.augment_minority_only = bool(augment_minority_only)
        self.minority_threshold = (
            minority_threshold
            if (minority_threshold is None or isinstance(minority_threshold, int))
            else None
        )
        self.k_per_record_minor = (
            int(k_per_record_minor) if k_per_record_minor is not None else None
        )

        # 统计口径：'median' 或 'mean'
        fs = str(fill_stat or "mean").lower()
        if fs not in ("median", "mean"):
            fs = "median"
        self.fill_stat = fs
        self._fill_stat_label = "中位数" if fs == "median" else "平均数"

        # 并发/限流
        self.reports_workers = max(1, int(reports_workers))
        self.fields_workers = fields_workers
        self.per_base_pool_maxsize = int(per_base_pool_maxsize)
        self.per_base_max_inflight = int(per_base_max_inflight)
        self.timeout = int(timeout)

        # 组合/结构
        self.compose_sep_prob = max(0.0, min(1.0, compose_sep_prob))
        self.compose_ocr_noise = bool(compose_ocr_noise)
        self.compose_paragraph_prob = max(0.0, min(1.0, float(compose_paragraph_prob)))
        self.value_linebreak_prob = max(0.0, min(1.0, float(value_linebreak_prob)))
        self.struct_perturb_enable = bool(struct_perturb_enable)
        self.struct_variants_per_record = max(0, int(struct_variants_per_record))
        self.struct_add_prob = max(0.0, min(1.0, float(struct_add_prob)))
        self.struct_drop_prob = max(0.0, min(1.0, float(struct_drop_prob)))
        self.struct_min_keys = max(1, int(struct_min_keys))

        self.exhaustive_titles = bool(exhaustive_titles)
        self.exhaustive_modules = bool(exhaustive_modules)
        self.synthesize_missing_titles = bool(synthesize_missing_titles)
        self.new_per_title = max(0, int(new_per_title))
        self.synth_keys_drop_prob = max(0.0, min(1.0, synth_keys_drop_prob))
        self.synth_min_keys = max(1, int(synth_min_keys))
        self.synth_max_keys = (
            int(synth_max_keys) if synth_max_keys is not None else None
        )
        self.synth_include_basic_info = bool(synth_include_basic_info)

        # 去重策略
        self.dedup_during_generation = bool(dedup_during_generation)
        self.final_dedup_mode = str(final_dedup_mode or "exact")
        self.final_sim_threshold = float(final_sim_threshold)
        self.final_sim_recent_k = max(50, int(final_sim_recent_k))

        self.flush_every = max(1, int(flush_every))
        self.synth_batch_size = max(1, int(synth_batch_size))
        self.inc_max_keys_per_record = (
            int(inc_max_keys_per_record)
            if inc_max_keys_per_record is not None
            else None
        )
        self.inc_key_pick = str(inc_key_pick or "head").lower()
        self.seed = int(seed)
        self.fields_synth_mode = str(fields_synth_mode or "batched").lower()
        
        # 新增：报告类型过滤和数量控制
        self.target_report_types = target_report_types
        self.max_samples_per_type = max_samples_per_type
        self.max_total_samples = max_total_samples

        # 组件（include_context_summary 透传到客户端）
        self.fieldwise_client = OpenAIFieldWiseLLM(
            model=openai_model,
            base_url=openai_base,
            api_key=api_key,
            max_workers=fields_workers or 8,
            timeout=self.timeout,
            rag_path=rag_path,
            rag_topk=rag_topk,
            per_base_pool_maxsize=self.per_base_pool_maxsize,
            per_base_max_inflight=self.per_base_max_inflight,
            include_context_summary=include_context_summary,
        )
        self.augmenter = LLMPromptAugmenter(
            tokenizer_name=tokenizer_name,
            seed=seed,
            fieldwise_client=self.fieldwise_client,
            compose_sep_prob=self.compose_sep_prob,
            compose_ocr_noise=self.compose_ocr_noise,
            compose_paragraph_prob=self.compose_paragraph_prob,
            value_linebreak_prob=self.value_linebreak_prob,
        )

        # 统计
        self._pre_stats: Optional[Dict[str, Any]] = None
        self._post_stats: Optional[Dict[str, Any]] = None
        self._minority_titles: set[str] = set()

        # JSONL 路径
        self._aug_jsonl = Path(f"{self.out_path}.aug.jsonl")
        # 抽样计数器：用于为每次字段子集抽样注入不同的随机种子
        self._jitter_counter: int = 0

    # ----------------------------- 结构辅助 -----------------------------
    def _synthesize_record(self, title: str, keys: List[str]) -> Dict[str, Any]:
        """调用后端：仅生成指定 keys 的值（我们只取这些键）。"""
        return self.fieldwise_client.synthesize_record(title, keys)

    def _target_from_counts(self, counts: List[int]) -> int:
        """根据 fill_stat 计算均衡目标（整数）。"""
        if not counts:
            return 0
        if self.fill_stat == "mean":
            return int(round(sum(counts) / max(1, len(counts))))
        # 默认中位数
        counts = sorted(counts)
        return int(counts[len(counts) // 2])

    def _synthesize_record_batched(
        self,
        title: str,
        keys: List[str],
        on_progress: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        为避免一次请求字段过多（JSON 超长/引号未闭合等导致解析失败），
        按 batch 切分多次合成并合并结果。
        """
        out: Dict[str, Any] = {"report_title": title}
        keys2 = [k for k in keys if isinstance(k, str) and k.strip()]
        bs = self.synth_batch_size
        for i in range(0, len(keys2), bs):
            part = keys2[i : i + bs]
            try:
                obj = self._synthesize_record(title, part)
                # 放宽：标题以后端为准（若有）
                if isinstance(obj, dict) and obj.get("report_title"):
                    out["report_title"] = str(obj.get("report_title")).strip() or title
                missing = []
                for k in part:
                    v = obj.get(k, "") if isinstance(obj, dict) else ""
                    if isinstance(v, str):
                        v = v.strip()
                    if v:
                        out[k] = v
                    else:
                        missing.append(k)
                    if on_progress:
                        try:
                            on_progress(1)
                        except Exception:
                            pass
                # 对于缺失字段，降级为逐字段补齐
                for k in missing:
                    try:
                        obj1 = self._synthesize_record(title, [k])
                        v1 = obj1.get(k, "") if isinstance(obj1, dict) else ""
                        if isinstance(v1, str):
                            v1 = v1.strip()
                        out[k] = v1
                    except Exception as e1:
                        print(f"[synth-batch:fallback] 失败 title='{title}' | key={k} | err={e1}")
                    finally:
                        if on_progress:
                            try:
                                on_progress(1)
                            except Exception:
                                pass
            except Exception as e:
                print(f"[synth-batch] 失败 title='{title}' | batch={part[:4]}… | err={e}")
                # 整批失败则逐字段尝试
                for k in part:
                    try:
                        obj1 = self._synthesize_record(title, [k])
                        v1 = obj1.get(k, "") if isinstance(obj1, dict) else ""
                        if isinstance(v1, str):
                            v1 = v1.strip()
                        out[k] = v1
                    except Exception as e1:
                        print(f"[synth-batch:fallback] 失败 title='{title}' | key={k} | err={e1}")
                    finally:
                        if on_progress:
                            try:
                                on_progress(1)
                            except Exception:
                                pass
        return out

    def _synthesize_fields_concurrent(
        self,
        title: str,
        keys: List[str],
        on_progress: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        逐字段并发合成：使用 fields_workers 控制并发度，每完成一个字段即更新进度。
        注意：对后端压力更大，请按需启用。
        """
        ks = [k for k in keys if isinstance(k, str) and k.strip()]
        out: Dict[str, Any] = {"report_title": title}
        if not ks:
            return out
        maxw = max(1, int(self.fields_workers or 8))
        import concurrent.futures as _f

        def task(k: str) -> Tuple[str, str]:
            try:
                obj = self._synthesize_record(title, [k])
                v = obj.get(k, "") if isinstance(obj, dict) else ""
                if isinstance(v, str):
                    v = v.strip()
                return k, v
            except Exception:
                return k, ""

        with _f.ThreadPoolExecutor(max_workers=maxw) as ex:
            futs = [ex.submit(task, k) for k in ks]
            for fut in _f.as_completed(futs):
                k, v = fut.result()
                out[k] = v
                if on_progress:
                    try:
                        on_progress(1)
                    except Exception:
                        pass
        return out

    def _limit_inc_keys(self, inc_keys: List[str]) -> List[str]:
        m = self.inc_max_keys_per_record
        if not m or m <= 0 or len(inc_keys) <= m:
            return inc_keys
        if self.inc_key_pick == "random":
            arr = list(inc_keys)
            random.Random(self.seed).shuffle(arr)
            return arr[:m]
        # 默认保留结构顺序的前 m 个
        return inc_keys[:m]

    def _limit_inc_keys_with_title(self, inc_keys: List[str], title: str) -> List[str]:
        """
        与 _limit_inc_keys 相同，但在随机抽样时混入标题以避免每条记录抽样一致。
        """
        m = self.inc_max_keys_per_record
        if not m or m <= 0 or len(inc_keys) <= m:
            return inc_keys
        if self.inc_key_pick == "random":
            arr = list(inc_keys)
            # 使用 (seed, title) 作为随机种子，确保不同标题或不同运行时更分散
            rnd = random.Random(hash((self.seed, title)) & 0xFFFFFFFF)
            rnd.shuffle(arr)
            return arr[:m]
        return inc_keys[:m]

    def _jitter_synth_keys(self, title: str, keys: List[str]) -> List[str]:
        """
        仅按丢弃概率对传入 keys 做子集选择，并保证不少于 synth_min_keys；
        不再对“基础信息类字段”做特殊处理。
        """
        ks = [k for k in keys if isinstance(k, str) and k.strip()]
        # 为每次调用注入不同种子：依赖 (seed, title, 调用序号)
        self._jitter_counter += 1
        rnd = random.Random(hash((self.seed, title, self._jitter_counter)) & 0xFFFFFFFF)

        # 丢弃概率抽样
        if self.synth_keys_drop_prob > 0.0:
            kept = [k for k in ks if rnd.random() >= self.synth_keys_drop_prob]
        else:
            kept = list(ks)

        # 至少保留 synth_min_keys
        if len(kept) < self.synth_min_keys:
            cand = [k for k in ks if k not in kept]
            rnd.shuffle(cand)
            kept += cand[: self.synth_min_keys - len(kept)]

        # 保持结构顺序输出
        seen, out = set(), []
        for k in ks:
            if k in kept and k not in seen:
                out.append(k)
                seen.add(k)

        # 额外硬上限：随机裁到 synth_max_keys（基于本次 rnd）
        if self.synth_max_keys is not None and len(out) > self.synth_max_keys:
            arr = list(out)
            rnd.shuffle(arr)
            out = arr[: self.synth_max_keys]
        return out

    def _apply_structural_perturbations(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not (self.struct_perturb_enable and self.struct_variants_per_record > 0):
            return records
        out = list(records)
        for r in records:
            title = str(r.get("report_title", "")).strip()
            map_keys = REPORT_STRUCTURE_MAP.get(title)
            if not map_keys:
                continue
            for _ in range(self.struct_variants_per_record):
                rr = {k: copy.deepcopy(v) for k, v in r.items() if k != "report"}
                cur = [k for k in rr.keys() if k not in RESERVED_KEYS]
                for k in [k for k in cur if k in set(map_keys)]:
                    if random.random() < self.struct_drop_prob:
                        rr.pop(k, None)
                for k in [k for k in map_keys if k not in rr]:
                    if random.random() < self.struct_add_prob:
                        rr[k] = ""
                kept = [k for k in rr.keys() if k not in RESERVED_KEYS]
                if len(kept) < self.struct_min_keys:
                    need = self.struct_min_keys - len(kept)
                    cand = [k for k in map_keys if k not in kept]
                    random.shuffle(cand)
                    for k in cand[:need]:
                        rr.setdefault(k, "")
                rr["report"] = self.augmenter.compose_report(rr, randomize=False)
                out.append(rr)
        return out

    # ----------------------------- 合并去重 -----------------------------
    def _write_final_merged(
        self, originals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """读取 JSONL 中的增强样本，与 originals 合并，并做去重。"""
        augmented: List[Dict[str, Any]] = []
        if self._aug_jsonl.exists():
            with self._aug_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            augmented.append(obj)
                    except Exception:
                        continue

        seen_keys = set()
        out: List[Dict[str, Any]] = []

        def _push_if_new(r: Dict[str, Any]) -> None:
            rep = str(r.get("report", "")).strip()
            title = str(r.get("report_title", "")).strip()
            if not rep or not title:
                return
            
            # 最终输出时也应用报告类型过滤
            if self.target_report_types is not None:
                if title not in self.target_report_types:
                    return  # 跳过不在目标类型中的报告
            
            key = (title, _md5(_norm_text_for_hash(rep)))
            if key not in seen_keys:
                out.append(r)
                seen_keys.add(key)

        for r in originals:
            _push_if_new(r)

        if self.final_dedup_mode == "exact":
            for r in augmented:
                _push_if_new(r)
        elif self.final_dedup_mode == "exact+similar":
            from difflib import SequenceMatcher

            recent_by_title: Dict[str, List[str]] = {}
            for r in out:
                t = str(r.get("report_title", "")).strip()
                rep = str(r.get("report", "")).strip()
                if t and rep:
                    bucket = recent_by_title.setdefault(t, [])
                    if len(bucket) < self.final_sim_recent_k:
                        bucket.append(rep)
            for r in augmented:
                rep = str(r.get("report", "")).strip()
                title = str(r.get("report_title", "")).strip()
                if not rep or not title:
                    continue
                key = (title, _md5(_norm_text_for_hash(rep)))
                if key in seen_keys:
                    continue
                bucket = recent_by_title.setdefault(title, [])
                is_dup = any(
                    SequenceMatcher(None, rep, s).ratio() >= self.final_sim_threshold
                    for s in bucket
                )
                if not is_dup:
                    out.append(r)
                    seen_keys.add(key)
                    bucket.append(rep)
                    if len(bucket) > self.final_sim_recent_k:
                        bucket.pop(0)

        # 原子写盘
        p = Path(self.out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

        # 仅返回“新增的增强样本”，避免 O(n^2) 的字典比较
        def _key(r: Dict[str, Any]):
            t = str(r.get("report_title", "")).strip()
            rep = str(r.get("report", "")).strip()
            if not t or not rep:
                return None
            return (t, _md5(_norm_text_for_hash(rep)))

        orig_keys = set()
        for r in originals:
            k = _key(r)
            if k:
                orig_keys.add(k)
        return [r for r in out if (_key(r) not in orig_keys)]

    # ----------------------------- 计算某条记录的“新增 key” -----------------------------
    def _inc_keys_for_record(self, rec: Dict[str, Any]) -> List[str]:
        title = str(rec.get("report_title") or rec.get("report_titles") or "").strip()
        map_keys = REPORT_STRUCTURE_MAP.get(title) or []
        exist = {k for k in rec.keys() if isinstance(k, str) and k not in RESERVED_KEYS}
        out: List[str] = []
        for k in map_keys:
            ks = str(k).strip()
            if not ks:
                continue
            if ks in exist:
                continue
            if ks in SYNTH_EXCLUDE_KEYS:
                continue
            # 规则排除：包含“号/号码/电话/类型/时间/日期”等的字段名
            if any(tok in ks for tok in ("号", "号码", "电话", "类型", "时间", "日期")):
                continue
            out.append(ks)
        return out

    # ----------------------------- 主流程 -----------------------------
    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        print(f"Loading raw dataset from {self.in_path} ...")
        raw: List[Dict[str, Any]] = json.loads(
            Path(self.in_path).read_text(encoding="utf-8")
        )

        # 统一标题键
        for r in raw:
            if not r.get("report_title") and r.get("report_titles"):
                r["report_title"] = r.get("report_titles")
                r.pop("report_titles", None)

        # 基础 report（不随机化，便于统计）+ 报告类型过滤
        base: List[Dict[str, Any]] = []
        type_counts: Dict[str, int] = {}
        filtered_by_type: Dict[str, List[Dict[str, Any]]] = {}
        
        # 第一步：过滤指定报告类型的数据
        for rec in raw:
            report_title = str(rec.get("report_title", "")).strip()
            
            # 报告类型过滤
            if self.target_report_types is not None:
                if report_title not in self.target_report_types:
                    continue  # 跳过不在目标类型中的报告
            
            if report_title not in filtered_by_type:
                filtered_by_type[report_title] = []
            filtered_by_type[report_title].append(rec)
        
        # 第二步：对每个报告类型检查数量，决定是否需要增强
        for report_title, records in filtered_by_type.items():
            current_count = len(records)
            target_count = self.max_samples_per_type if self.max_samples_per_type is not None else current_count
            
            print(f"[AUGMENT] {report_title}: 现有{current_count}个样本，目标{target_count}个")
            
            if current_count < target_count:
                # 需要增强
                need_more = target_count - current_count
                print(f"[AUGMENT] {report_title}: 需要增强{need_more}个样本")
                
                # 计算每个原始样本需要增强多少次
                augment_per_record = max(1, (need_more + current_count - 1) // current_count)
                print(f"[AUGMENT] {report_title}: 每个原样本增强{augment_per_record}次")
                
                # 添加原始数据
                print(f"[PROCESS] {report_title}: 处理{len(records)}个原始样本")
                pbar_orig = tqdm(total=len(records), desc=f"处理{report_title}原始数据", ncols=100, mininterval=0.1)
                for rec in records:
                    if len(base) >= (self.max_total_samples or float('inf')):
                        break
                    r = copy.deepcopy(rec)
                    r["report"] = self.augmenter.compose_report(r, randomize=False)
                    base.append(r)
                    type_counts[report_title] = type_counts.get(report_title, 0) + 1
                    pbar_orig.update(1)
                pbar_orig.close()
                
                # 添加增强数据 - 使用并发处理
                added_count = 0
                # 创建增强进度条
                pbar_aug = tqdm(total=need_more, desc=f"增强{report_title}", ncols=100, mininterval=0.2)
                
                # 准备增强任务列表
                augment_tasks = []
                for rec in records:
                    for _ in range(min(augment_per_record, need_more - len(augment_tasks))):
                        if len(augment_tasks) >= need_more:
                            break
                        augment_tasks.append(rec)
                    if len(augment_tasks) >= need_more:
                        break
                
                print(f"[AUGMENT] {report_title}: 使用{self.reports_workers}个并发worker处理{len(augment_tasks)}个增强任务")
                
                # 使用 reports_workers 进行并发增强
                import concurrent.futures
                max_workers = max(1, min(self.reports_workers, len(augment_tasks)))
                
                def augment_task(rec):
                    try:
                        augmented = self.augmenter.augment_once(rec)
                        augmented["report"] = self.augmenter.compose_report(augmented, randomize=True)
                        return augmented
                    except Exception as e:
                        print(f"[AUGMENT] 单个增强失败: {report_title} - {e}")
                        return None
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有增强任务
                    future_to_task = {executor.submit(augment_task, rec): rec for rec in augment_tasks}
                    
                    # 收集完成的结果
                    for future in concurrent.futures.as_completed(future_to_task):
                        if added_count >= need_more:
                            break
                        if len(base) >= (self.max_total_samples or float('inf')):
                            break
                            
                        result = future.result()
                        if result is not None:
                            base.append(result)
                            type_counts[report_title] = type_counts.get(report_title, 0) + 1
                            added_count += 1
                        
                        pbar_aug.update(1)  # 更新进度条
                
                pbar_aug.close()  # 关闭进度条
                print(f"[AUGMENT] {report_title}: 成功增强{added_count}个样本（并发度：{max_workers}）")
            else:
                # 不需要增强，直接使用现有数据（可能需要截断）
                use_count = min(current_count, target_count)
                print(f"[PROCESS] {report_title}: 处理{use_count}个样本（无需增强）")
                pbar_direct = tqdm(total=use_count, desc=f"处理{report_title}", ncols=100, mininterval=0.1)
                for i, rec in enumerate(records[:use_count]):
                    if len(base) >= (self.max_total_samples or float('inf')):
                        break
                    r = copy.deepcopy(rec)
                    r["report"] = self.augmenter.compose_report(r, randomize=False)
                    base.append(r)
                    type_counts[report_title] = type_counts.get(report_title, 0) + 1
                    pbar_direct.update(1)
                pbar_direct.close()
        
        # 打印过滤统计
        if self.target_report_types is not None:
            print(f"[FILTER] 目标报告类型: {self.target_report_types}")
        if self.max_samples_per_type is not None:
            print(f"[FILTER] 每类型最大样本数: {self.max_samples_per_type}")
        if self.max_total_samples is not None:
            print(f"[FILTER] 总样本数上限: {self.max_total_samples}")
        print(f"[FILTER] 过滤后样本数: {len(base)} (原始: {len(raw)})")
        if type_counts:
            print(f"[FILTER] 各类型样本数: {dict(sorted(type_counts.items()))}")

        print("Computing pre-stats ...")
        self._pre_stats = self.augmenter.dataset_stats(base)
        print(f"Pre-stats: {self._pre_stats}")

        # ========== A0) 组合模式：增量新键 + 新类补样到目标均衡值 ==========
        if self.inc_and_synthesize_missing_to_median:
            print(f"[combo] 进行：现有记录补齐新增键 + 新增类别合成到{self._fill_stat_label}")

            # Ⅰ) 先对现有记录补齐“相对结构映射新增”的 key
            patched: List[Dict[str, Any]] = []
            total_with_inc = 0
            total_inc_keys = 0
            pbar_inc = tqdm(total=len(raw), desc="补新增字段", ncols=90, mininterval=0.2)
            if self.reports_workers and self.reports_workers > 1:
                import concurrent.futures as _f

                def _job_inc(rec: Dict[str, Any]):
                    title = str(rec.get("report_title", "")).strip()
                    inc_keys = self._limit_inc_keys_with_title(
                        self._inc_keys_for_record(rec), title
                    )
                    rr = copy.deepcopy(rec)
                    rr["added_keys"] = inc_keys
                    if not inc_keys:
                        return rr, 0
                    try:
                        if self.fields_synth_mode == "per_key":
                            synth = self._synthesize_fields_concurrent(title, inc_keys)
                        else:
                            synth = self._synthesize_record_batched(title, inc_keys)
                        for k in inc_keys:
                            vv = synth.get(k, "") if isinstance(synth, dict) else ""
                            if isinstance(vv, str) and vv.strip():
                                rr[k] = vv.strip()
                        return rr, len(inc_keys)
                    except Exception as e:
                        print(f"[combo:inc] 合成失败 title='{title}' | keys={inc_keys} | err={e}")
                        return rr, 0

                with _f.ThreadPoolExecutor(max_workers=self.reports_workers) as ex:
                    futs = [ex.submit(_job_inc, rec) for rec in raw]
                    for fut in _f.as_completed(futs):
                        rr, added = fut.result()
                        patched.append(rr)
                        if added > 0:
                            total_with_inc += 1
                            total_inc_keys += added
                        pbar_inc.update(1)
                pbar_inc.close()
            else:
                for rec in raw:
                    title = str(rec.get("report_title", "")).strip()
                    inc_keys = self._limit_inc_keys_with_title(
                        self._inc_keys_for_record(rec), title
                    )
                    if not inc_keys:
                        rr = copy.deepcopy(rec)
                        rr["added_keys"] = []
                        patched.append(rr)
                        pbar_inc.update(1)
                        continue

                    total_with_inc += 1
                    total_inc_keys += len(inc_keys)

                    rr = copy.deepcopy(rec)
                    rr["added_keys"] = inc_keys
                    # 字段层进度条
                    pbar_fields = tqdm(
                        total=len(inc_keys),
                        desc=f"字段@{title}",
                        ncols=90,
                        mininterval=0.1,
                        leave=False,
                    )
                    try:
                        if self.fields_synth_mode == "per_key":
                            synth = self._synthesize_fields_concurrent(
                                title, inc_keys, on_progress=pbar_fields.update
                            )
                        else:
                            synth = self._synthesize_record_batched(
                                title, inc_keys, on_progress=pbar_fields.update
                            )
                    except Exception as e:
                        print(
                            f"[combo:inc] 合成失败 title='{title}' | keys={inc_keys} | err={e}"
                        )
                        patched.append(rr)
                        # 失败也推进到下一个样本
                    finally:
                        try:
                            pbar_fields.close()
                        except Exception:
                            pass
                        if 'synth' not in locals():
                            continue
                    
                    # 调试输出：显示模型生成的字段内容
                    all_inc_keys = self._inc_keys_for_record(rec)
                    print(f"\n[新增字段] {title} - {rec.get('姓名', 'N/A')}")
                    print(f"  全部缺失字段({len(all_inc_keys)}个): {all_inc_keys}")
                    print(f"  随机选择字段({len(inc_keys)}个): {inc_keys}")
                    generated_fields = {}
                    
                    for k in inc_keys:
                        if k in synth and isinstance(synth[k], str) and synth[k].strip():
                            rr[k] = synth[k].strip()
                            generated_fields[k] = synth[k].strip()
                            print(f"  {k}: {synth[k].strip()}")
                    
                    if generated_fields:
                        print(f"  成功生成 {len(generated_fields)} 个字段")
                    else:
                        print(f"  未生成任何字段内容")
                    print("-" * 50)
                    
                    patched.append(rr)
                    pbar_inc.update(1)
                pbar_inc.close()

            # Ⅱ) 统计现有（补齐后）每个标题的数量，计算目标值（由 fill_stat 决定）
            counts_by_title: Dict[str, int] = {}
            for r in patched:
                t = str(r.get("report_title", "")).strip()
                if t:
                    counts_by_title[t] = counts_by_title.get(t, 0) + 1
            vals = list(counts_by_title.values())
            target = self._target_from_counts(vals)
            print(f"[combo] 现有类别数量{self._fill_stat_label} = {target}")

            # Ⅲ) 为缺失标题按目标值合成样本
            existing_titles = set(counts_by_title.keys())
            missing = [t for t in REPORT_STRUCTURE_MAP.keys() if t not in existing_titles]
            total_miss = sum(max(0, int(target)) for _ in missing)
            print(
                f"[combo] 缺失类别数={len(missing)}，计划合成总样本={total_miss}（每类至{self._fill_stat_label}）"
            )
            pbar_miss = tqdm(total=total_miss, desc="补缺失类别", ncols=90, mininterval=0.2)
            if self.reports_workers and self.reports_workers > 1:
                import concurrent.futures as _f
                # 为保证每个合成样本的字段子集不同，这里仅传递 title，
                # 在工作线程内部重新调用 _jitter_synth_keys() 进行随机抽样。
                jobs: List[Tuple[str, int]] = []
                for title in missing:
                    need = max(0, int(target))
                    if need <= 0:
                        continue
                    for _ in range(need):
                        jobs.append((title, 1))

                def _job_miss(task: Tuple[str, int]):
                    title, _ = task
                    # 每条样本独立抽样字段，避免固定为同一子集
                    keys = self._jitter_synth_keys(title, REPORT_STRUCTURE_MAP.get(title, []))
                    try:
                        if self.fields_synth_mode == "per_key":
                            rec = self._synthesize_fields_concurrent(title, keys)
                        else:
                            rec = self._synthesize_record_batched(title, keys)
                            # 强制保持原始 report_title（防止 LLM 返回带后缀的完整名称）
                            rec["report_title"] = title
                        rec["report"] = self.augmenter.compose_report(rec)
                        return rec
                    except Exception as e:
                        print(f"[combo:miss] 整条合成失败 title='{title}' | err={e}")
                        time.sleep(3)
                        return None

                with _f.ThreadPoolExecutor(max_workers=self.reports_workers) as ex:
                    futs = [ex.submit(_job_miss, t) for t in jobs]
                    for fut in _f.as_completed(futs):
                        r = fut.result()
                        if isinstance(r, dict):
                            patched.append(r)
                        pbar_miss.update(1)
                pbar_miss.close()
            else:
                for title in missing:
                    need = max(0, int(target))
                    if need <= 0:
                        continue
                    for j in range(need):
                        # 每条样本独立抽样字段，避免固定为同一子集
                        keys = self._jitter_synth_keys(title, REPORT_STRUCTURE_MAP.get(title, []))
                        # 字段层进度条（对整条合成改用分批接口，以便字段层进度）
                        pbar_fields = tqdm(
                            total=len(keys), desc=f"{title}[{j+1}/{need}] 字段", ncols=90, mininterval=0.1, leave=False
                        )
                        try:
                            if self.fields_synth_mode == "per_key":
                                rec = self._synthesize_fields_concurrent(
                                    title, keys, on_progress=pbar_fields.update
                                )
                            else:
                                rec = self._synthesize_record_batched(
                                    title, keys, on_progress=pbar_fields.update
                        )
                            # 强制保持原始 report_title（防止 LLM 返回带后缀的完整名称）
                            rec["report_title"] = title
                        except Exception as e:
                            print(f"[combo:miss] 整条合成失败 title='{title}' | err={e} | 跳过并等待 3s")
                            time.sleep(3)
                            pbar_miss.update(1)
                            continue
                        finally:
                            try:
                                pbar_fields.close()
                            except Exception:
                                pass
                        rec["report"] = self.augmenter.compose_report(rec)
                        patched.append(rec)
                        pbar_miss.update(1)
                pbar_miss.close()

            # Ⅳ) 直接输出合并后的数据集
            p = Path(self.out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(patched, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)

            print(
                f"[combo] records_with_inc={total_with_inc} | total_new_keys={total_inc_keys} | final={len(patched)}"
            )
            print(f"[combo] dataset written to: {self.out_path}")
            print(f"Done in {time.time()-t0:.2f}s")
            return {
                "pre": self._pre_stats,
                "post": None,
                "final": len(patched),
                "inc_records": total_with_inc,
                "inc_total_keys": total_inc_keys,
                # 目标均衡值（按 fill_stat 计算的中位数或平均数）
                "median": int(target),
                "new_titles": len(missing),
            }

        # ========== A) 仅做"为新增 key 合成值 + 写到 added_keys" ==========
        if self.inc_synthesize_new_keys and not self.inc_and_synthesize_missing_to_median:
            patched: List[Dict[str, Any]] = []
            total_with_inc = 0
            total_inc_keys = 0
            pbar_inc = tqdm(total=len(raw), desc="补新增字段", ncols=90, mininterval=0.2)
            if self.reports_workers and self.reports_workers > 1:
                import concurrent.futures as _f

                def _job_inc(rec: Dict[str, Any]):
                    title = str(rec.get("report_title", "")).strip()
                    inc_keys = self._limit_inc_keys_with_title(
                        self._inc_keys_for_record(rec), title
                    )
                    rr = copy.deepcopy(rec)
                    rr["added_keys"] = inc_keys
                    if not inc_keys:
                        return rr, 0
                    try:
                        if self.fields_synth_mode == "per_key":
                            synth = self._synthesize_fields_concurrent(title, inc_keys)
                        else:
                            synth = self._synthesize_record_batched(title, inc_keys)
                        
                        # 调试输出：显示模型生成的字段内容
                        all_inc_keys = self._inc_keys_for_record(rec)
                        print(f"\n[新增字段] {title} - {rec.get('姓名', 'N/A')}")
                        print(f"  全部缺失字段({len(all_inc_keys)}个): {all_inc_keys}")
                        print(f"  随机选择字段({len(inc_keys)}个): {inc_keys}")
                        generated_fields = {}
                        
                        for k in inc_keys:
                            vv = synth.get(k, "") if isinstance(synth, dict) else ""
                            if isinstance(vv, str) and vv.strip():
                                rr[k] = vv.strip()
                                generated_fields[k] = vv.strip()
                                print(f"  {k}: {vv.strip()}")
                        
                        if generated_fields:
                            print(f"  成功生成 {len(generated_fields)} 个字段")
                        else:
                            print(f"  未生成任何字段内容")
                        print("-" * 50)
                        
                        return rr, len(inc_keys)
                    except Exception as e:
                        print(f"[inc-synth] 合成失败 title='{title}' | keys={inc_keys} | err={e}")
                        return rr, 0

                with _f.ThreadPoolExecutor(max_workers=self.reports_workers) as ex:
                    futs = [ex.submit(_job_inc, rec) for rec in raw]
                    for fut in _f.as_completed(futs):
                        rr, added = fut.result()
                        patched.append(rr)
                        if added > 0:
                            total_with_inc += 1
                            total_inc_keys += added
                        pbar_inc.update(1)
                pbar_inc.close()
            else:
                for rec in raw:
                    title = str(rec.get("report_title", "")).strip()
                    inc_keys = self._limit_inc_keys_with_title(
                        self._inc_keys_for_record(rec), title
                    )
                    if not inc_keys:
                        rr = copy.deepcopy(rec)
                        rr["added_keys"] = []
                        patched.append(rr)
                        pbar_inc.update(1)
                        continue

                    total_with_inc += 1
                    total_inc_keys += len(inc_keys)

                    rr = copy.deepcopy(rec)
                    rr["added_keys"] = inc_keys
                    # 字段层进度条
                    pbar_fields = tqdm(
                        total=len(inc_keys),
                        desc=f"字段@{title}",
                        ncols=90,
                        mininterval=0.1,
                        leave=False,
                    )
                    try:
                        if self.fields_synth_mode == "per_key":
                            synth = self._synthesize_fields_concurrent(
                                title, inc_keys, on_progress=pbar_fields.update
                            )
                        else:
                            synth = self._synthesize_record_batched(
                                title, inc_keys, on_progress=pbar_fields.update
                            )
                    except Exception as e:
                        print(
                            f"[inc-synth] 合成失败 title='{title}' | keys={inc_keys} | err={e}"
                        )
                        patched.append(rr)
                        # 失败也推进到下一个样本
                    finally:
                        try:
                            pbar_fields.close()
                        except Exception:
                            pass
                        if 'synth' not in locals():
                            continue
                    
                    # 调试输出：显示模型生成的字段内容
                    all_inc_keys = self._inc_keys_for_record(rec)
                    print(f"\n[新增字段] {title} - {rec.get('姓名', 'N/A')}")
                    print(f"  全部缺失字段({len(all_inc_keys)}个): {all_inc_keys}")
                    print(f"  随机选择字段({len(inc_keys)}个): {inc_keys}")
                    generated_fields = {}
                    
                    for k in inc_keys:
                        if k in synth and isinstance(synth[k], str) and synth[k].strip():
                            rr[k] = synth[k].strip()
                            generated_fields[k] = synth[k].strip()
                            print(f"  {k}: {synth[k].strip()}")
                    
                    if generated_fields:
                        print(f"  成功生成 {len(generated_fields)} 个字段")
                    else:
                        print(f"  未生成任何字段内容")
                    print("-" * 50)
                    
                    patched.append(rr)
                    pbar_inc.update(1)
                pbar_inc.close()

            # 如果只需要字段补齐，则直接返回
            if not (self.topk_to_median or self.k_per_record > 0 or self.struct_perturb_enable or self.synthesize_missing_titles):
                # 直接输出补齐后的数据集；不改 report 文本（保持稳定）
                p = Path(self.out_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(p.suffix + ".tmp")
                tmp.write_text(
                    json.dumps(patched, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                tmp.replace(p)

                print(
                    f"[inc-synth] records_with_inc={total_with_inc} | total_new_keys={total_inc_keys}"
                )
                print(f"[inc-synth] patched dataset written to: {self.out_path}")
                print(f"Done in {time.time()-t0:.2f}s")
                return {
                    "pre": self._pre_stats,
                    "post": None,
                    "final": len(patched),
                    "inc_records": total_with_inc,
                    "inc_total_keys": total_inc_keys,
                }
            
            # 如果需要数据增强，则继续执行增强流程
            # 更新raw为patched，继续执行后续增强流程
            raw = patched
            base = []
            for rec in raw:
                # 清理已从结构映射中删除的字段
                title = str(rec.get("report_title", "")).strip()
                valid_keys = set(REPORT_STRUCTURE_MAP.get(title, []))
                # 添加保留字段
                valid_keys.update(RESERVED_KEYS)
                # 如果存在added_keys，也保留这些字段
                if "added_keys" in rec and isinstance(rec["added_keys"], list):
                    valid_keys.update(rec["added_keys"])
                
                # 过滤掉已从结构映射中删除的字段
                filtered_rec = {}
                for k, v in rec.items():
                    # 保留所有有效字段和保留字段
                    if k in valid_keys:
                        filtered_rec[k] = v
                    # 特殊处理added_keys字段，确保同步
                    elif k == "added_keys" and isinstance(v, list):
                        # 过滤掉已删除的字段
                        filtered_keys = [key for key in v if key in REPORT_STRUCTURE_MAP.get(title, [])]
                        filtered_rec[k] = filtered_keys
                
                r = copy.deepcopy(filtered_rec)
                r["report"] = self.augmenter.compose_report(r, randomize=False)
                base.append(r)

        # ========== B) 正常增强流程（默认关闭；保留能力） ==========
        # 原始分布（用于 topk_to_median / 少数类策略）
        raw_counts_by_title: Dict[str, int] = {}
        for r in raw:
            t_norm = _normalize_title_for_counting(r.get("report_title") or "")
            if t_norm:
                raw_counts_by_title[t_norm] = raw_counts_by_title.get(t_norm, 0) + 1

        # 少数类集合（旧模式用，默认关闭）
        self._minority_titles = set()
        if self.augment_minority_only and raw_counts_by_title:
            vals = sorted(raw_counts_by_title.values())
            mid = len(vals) // 2
            thresh = (
                vals[mid]
                if self.minority_threshold is None
                else int(self.minority_threshold)
            )
            self._minority_titles = {
                t for t, c in raw_counts_by_title.items() if c <= thresh
            }
            print(
                f"[minority:init] titles={len(self._minority_titles)} threshold={thresh} raw_total={sum(raw_counts_by_title.values())}"
            )

        # JSONL：清空旧文件 + 启动异步写
        try:
            aug_jsonl = Path(f"{self.out_path}.aug.jsonl")
            if aug_jsonl.exists():
                aug_jsonl.unlink()
        except Exception:
            pass
        saver = _AsyncSaver(Path(f"{self.out_path}.aug.jsonl"))
        saver.start()

        # 计划：A) topk_to_median；B) old-mode
        if self.topk_to_median and raw_counts_by_title:
            vals = list(raw_counts_by_title.values())
            target = self._target_from_counts(vals)

            sorted_titles = sorted(raw_counts_by_title.items(), key=lambda x: x[1])
            picked_norms: List[str] = []
            for norm, cnt in sorted_titles:
                if cnt < target:
                    picked_norms.append(norm)
                if len(picked_norms) >= self.topk_titles:
                    break

            needs: Dict[str, int] = {}
            for norm in picked_norms:
                cnt = raw_counts_by_title.get(norm, 0)
                need = max(0, target - cnt)
                if need > 0:
                    needs[norm] = need

            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for rec in base:
                norm = _normalize_title_for_counting(rec.get("report_title", ""))
                if norm in needs:
                    buckets.setdefault(norm, []).append(rec)

            work_items: List[Dict[str, Any]] = []
            for norm, need in needs.items():
                arr = buckets.get(norm, [])
                if not arr:
                    continue
                n = len(arr)
                q, r = divmod(need, n)
                arr2 = list(arr)
                random.shuffle(arr2)
                for i, rec in enumerate(arr2):
                    attempts = q + (1 if i < r else 0)
                    for _ in range(attempts):
                        work_items.append(rec)

            print(
                f"[topk-balance:init] {self._fill_stat_label}={target} picked={len(picked_norms)} needs={json.dumps(needs, ensure_ascii=False)}"
            )
            print(f"[topk-balance:plan] total_need={len(work_items)}")

        else:
            # 旧模式：按标题 attempts
            work = self._apply_structural_perturbations(base)

            if self.synthesize_missing_titles:
                existing = {
                    str(r.get("report_title", "")).strip()
                    for r in work
                    if r.get("report_title")
                }
                missing = [t for t in REPORT_STRUCTURE_MAP.keys() if t not in existing]
                total_miss = len(missing) * max(1, self.new_per_title)
                idx_miss = 0
                for title in missing:
                    keys = self._jitter_synth_keys(
                        title, REPORT_STRUCTURE_MAP.get(title, [])
                    )
                    for j in range(self.new_per_title):
                        idx_miss += 1
                        print(
                            f"合成进度 {idx_miss}/{total_miss}：{title} ({j+1}/{self.new_per_title})"
                        )
                        try:
                            rec = self._synthesize_record(title, keys)
                        except Exception as e:
                            print(
                                f"[synthesize] 整条合成失败 title='{title}' | err={e} | 跳过并等待 3s"
                            )
                            time.sleep(3)
                            continue
                        rec["report"] = self.augmenter.compose_report(rec)
                        work.append(rec)

            def _attempts_for_title(title_raw: str) -> int:
                if self.augment_minority_only:
                    t_norm = _normalize_title_for_counting(title_raw)
                    if t_norm not in self._minority_titles:
                        return 0
                    if self.k_per_record_minor is not None:
                        return max(0, int(self.k_per_record_minor))
                return max(0, int(self.k_per_record))

            if self.augment_minority_only and self._minority_titles:
                before = len(work)
                work = [
                    w
                    for w in work
                    if _normalize_title_for_counting(str(w.get("report_title", "")))
                    in self._minority_titles
                ]
                print(
                    f"[minority:filter] titles={len(self._minority_titles)} work: {before}->{len(work)}"
                )

            work_items: List[Dict[str, Any]] = []
            for rec in work:
                attempts = _attempts_for_title(str(rec.get("report_title", "")).strip())
                for _ in range(attempts):
                    work_items.append(rec)
            print(f"增强总数：{len(work_items)}")

        # 生成阶段：不做任何相似度/去重，直接落 JSONL
        def process(item: Tuple[int, Dict[str, Any]]):
            i, rec = item
            try:
                aug = self.augmenter.augment_once(rec)
            except Exception as e:
                print(
                    f"[process] 样本增强失败 idx={i} title='{rec.get('report_title','')}' | err={e}"
                )
                return
            
            # 清理已从结构映射中删除的字段
            # 获取当前report_title对应的结构映射字段列表
            title = str(aug.get("report_title", "")).strip()
            valid_keys = set(REPORT_STRUCTURE_MAP.get(title, []))
            # 添加保留字段
            valid_keys.update(RESERVED_KEYS)
            # 如果存在added_keys，也保留这些字段
            if "added_keys" in aug and isinstance(aug["added_keys"], list):
                valid_keys.update(aug["added_keys"])
            
            # 过滤掉已从结构映射中删除的字段
            filtered_aug = {}
            for k, v in aug.items():
                # 保留所有有效字段和保留字段
                if k in valid_keys:
                    filtered_aug[k] = v
                # 特殊处理added_keys字段，确保同步
                elif k == "added_keys" and isinstance(v, list):
                    # 过滤掉已删除的字段
                    filtered_keys = [key for key in v if key in REPORT_STRUCTURE_MAP.get(title, [])]
                    filtered_aug[k] = filtered_keys
            
            # 确保added_keys与实际新增字段同步
            # 如果原记录有added_keys，则增强后的记录也应该保持同步
            if "added_keys" in rec and isinstance(rec["added_keys"], list):
                # 过滤掉增强后为空值的字段
                synced_added_keys = [
                    k for k in rec["added_keys"] 
                    if k in filtered_aug and isinstance(filtered_aug[k], str) and filtered_aug[k].strip()
                ]
                filtered_aug["added_keys"] = synced_added_keys
            elif "added_keys" in filtered_aug:
                # 如果增强过程中产生了added_keys，但原记录没有，则保留增强后的added_keys
                pass
            # 如果都没有added_keys，则不添加
                
            saver.enqueue([filtered_aug])

        total = locals().get("work_items", [])
        print("Starting augmentation (no-dedup, streaming to JSONL) ...")
        pbar = tqdm(total=len(total), desc="增强进度", ncols=90, mininterval=0.2)

        try:
            if not total:
                pass
            elif self.reports_workers <= 1:
                for i, rec in enumerate(total):
                    process((i, rec))
                    pbar.update(1)
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.reports_workers
                ) as ex:
                    for _ in ex.map(process, enumerate(total)):
                        pbar.update(1)
        finally:
            pbar.close()
            saver.stop()

        # 最终合并（只此处去重）
        augmented = self._write_final_merged(base)
        self._post_stats = self.augmenter.dataset_stats(augmented)
        print(f"Post-stats (aug only): {self._post_stats}")
        print(f"Done in {time.time()-t0:.2f}s")

        # 统计最终输出数据总数（out_path 里的合并结果）
        try:
            final_total = len(
                json.loads(Path(self.out_path).read_text(encoding="utf-8"))
            )
        except Exception as e:
            print(f"[final-count] 读取 {self.out_path} 失败: {e}")
            final_total = 0

        print(f"Final total samples: {final_total}")
        return {"pre": self._pre_stats, "post": self._post_stats, "final": final_total}
