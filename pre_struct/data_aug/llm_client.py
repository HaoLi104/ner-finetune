# llm_client.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter

from utils import _truncate
from structs import REPORT_STRUCTURE_MAP, get_key_alias_maps
from rag import get_cached_rag_index


class OpenAIFieldWiseLLM:
    """
    多后端并发负载均衡客户端（修复版）：
    - base_url 支持 "A|B|C" 或 list[str]，轮询分发到多个后端
    - 每个后端独立连接池 + 并发闸门（Semaphore）限制在飞请求
    - 所有后端忙时可选择进入等待队列（queue_when_busy=True）
    - 对 5xx/网络异常做有限次重试（指数退避）
    - 其余接口/行为保持与原版一致
    """

    def __init__(
        self,
        model: str = "qwen3-32b",
        base_url: Union[str, List[str]] = "https://qwen3.yoo.la/v1/",
        api_key: Optional[str] = None,
        max_workers: int = 12,
        timeout: int = 20,
        rag_path: Optional[str] = None,
        rag_topk: int = 2,
        # 并发与连接池
        per_base_pool_maxsize: int = 256,
        per_base_max_inflight: int = 64,
        include_context_summary: bool = True,
        # 忙时排队与重试
        queue_when_busy: bool = True,
        busy_wait_timeout: float = 30.0,  # 所有后端都忙时的单次等待上限（秒）
        max_retries: int = 2,  # 出错重试次数（不含首轮）
        retry_backoff_base: float = 0.6,  # 重试指数退避基础（秒）
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_workers = int(max_workers)
        self.timeout = int(timeout)
        self.include_context_summary = bool(include_context_summary)
        self.model = str(model)

        # -------- 修复点 1：规范化 base_url，支持"|"拆分 --------
        if isinstance(base_url, str):
            bases = [b.strip().rstrip("/") for b in base_url.split("|") if b.strip()]
        else:
            bases = [str(b).strip().rstrip("/") for b in base_url if str(b).strip()]
        if not bases:
            raise RuntimeError("No backend configured")
        self._base_urls = bases
        print(f"[LLM:init] backends = {self._base_urls}")

        # -------- 后端会话与并发闸门 --------
        self.backends: List[Dict[str, Any]] = []
        for bu in bases:
            sess = requests.Session()
            sess.headers.update(
                {"Accept": "application/json", "Content-Type": "application/json"}
            )
            if self.api_key:
                sess.headers.update({"Authorization": f"Bearer {self.api_key}"})
            adapter = HTTPAdapter(
                pool_connections=per_base_pool_maxsize,
                pool_maxsize=per_base_pool_maxsize,
                max_retries=0,  # 不在适配器层重试，交给下方逻辑
            )
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            self.backends.append(
                {
                    "session": sess,
                    "url": f"{bu}/model/single_report",
                    "sem": threading.Semaphore(int(per_base_max_inflight)),
                }
            )

        # 轮询指针
        self._rr_idx = random.randint(0, len(self.backends) - 1)
        self._rr_lock = threading.Lock()

        # 忙时排队与重试参数
        self.queue_when_busy = bool(queue_when_busy)
        self.busy_wait_timeout = float(busy_wait_timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff_base = float(retry_backoff_base)

        # -------- RAG 初始化（保持原行为） --------
        self.rag_topk = int(rag_topk)
        if rag_path and os.path.exists(rag_path):
            print(f"[RAG:init] rag_path={rag_path}")
            try:
                self.rag_index = (
                    get_cached_rag_index(
                        rag_path,
                        chunk_size=300,
                        chunk_overlap=30,
                        embedding_model=(
                            "/mnt/windows/Users/Admin/LLM/models/Qwen3-Embedding-0.6B/Qwen/Qwen3-Embedding-0.6B/"
                        ),
                        encode_batch_size=128,
                        show_progress=True,
                        use_fp16=True,
                    )
                    if (rag_path and os.path.exists(rag_path))
                    else None
                )
                if self.rag_index and self.rag_index.ready():
                    print(
                        f"[RAG:init] ready, chunks={len(self.rag_index.chunks)}, topk={self.rag_topk}"
                    )
                else:
                    print(f"[RAG:init] NOT ready (unexpected).")
            except Exception as e:
                print(f"[RAG:init] FAILED -> {e}")
                self.rag_index = None
        else:
            self.rag_index = None
            print(f"[RAG:init] disabled (rag_path missing or empty): {rag_path}")

        self._rag_warned = False

    # ----------------------------- 负载均衡 + 忙时排队 + 简易重试 -----------------------------
    def _post_report(self, body: Dict[str, Any]) -> Any:
        """
        轮询选择后端：
        1) 先快速尝试“非阻塞”获取任一后端的信号量（尽可能不排队，直接分流给空闲后端）；
        2) 若全忙且 queue_when_busy=True：对当前轮询指向的后端做一次“带超时”的 acquire 等待；
        3) 请求失败（网络/5xx）做有限次重试（对下一个后端，指数退避）。
        """
        n = len(self.backends)
        last_exc: Optional[BaseException] = None

        # 轮询起点
        with self._rr_lock:
            start_idx = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % n

        # ---------- 快速路径：非阻塞尝试获取任一后端 ----------
        for j in range(n):
            idx = (start_idx + j) % n
            b = self.backends[idx]
            if b["sem"].acquire(blocking=False):
                try:
                    return self._request_with_retries(b, body)
                finally:
                    b["sem"].release()

        # ---------- 全部忙：按需等待当前轮询指向后端 ----------
        if self.queue_when_busy:
            idx = start_idx  # 当前指向
            b = self.backends[idx]
            ok = b["sem"].acquire(timeout=self.busy_wait_timeout)
            if not ok:
                raise RuntimeError("All backends are busy (wait timeout)")
            try:
                return self._request_with_retries(b, body)
            finally:
                b["sem"].release()

        # 不排队直接报错
        raise RuntimeError("All backends are busy (no slot available)")

    def _request_with_retries(
        self, backend: Dict[str, Any], body: Dict[str, Any]
    ) -> Any:
        """
        对单个后端做有限次重试（仅在网络异常或 5xx 时重试，4xx 不重试）。
        每次重试切换到“下一个”后端（避免一直锚定一个坏节点）。
        """
        n = len(self.backends)
        # 找到当前后端索引
        try:
            base_idx = self.backends.index(backend)
        except ValueError:
            base_idx = 0

        for attempt in range(0, self.max_retries + 1):
            idx = (base_idx + attempt) % n
            b = self.backends[idx]
            # 二次并发闸门：如果不是第 0 次（首轮已经拿到 sem 了），此处需要再拿一次
            need_acquire = attempt > 0
            acquired = False
            if need_acquire:
                acquired = b["sem"].acquire(timeout=self.busy_wait_timeout)
                if not acquired:
                    # 当前后端仍忙，进入下一个后端重试
                    continue
            try:
                resp = b["session"].post(b["url"], json=body, timeout=self.timeout)
                if 200 <= resp.status_code < 300:
                    try:
                        return resp.json()
                    except Exception as e:
                        raise RuntimeError(f"Backend non-JSON: {e}")
                # 非 2xx
                if 500 <= resp.status_code < 600:
                    # 5xx：可重试
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff_base * (2**attempt))
                        continue
                    raise RuntimeError(
                        f"HTTP {resp.status_code} @ {b['url']}: {resp.text[:200]}"
                    )
                else:
                    # 4xx：不重试，直接抛错
                    raise RuntimeError(
                        f"HTTP {resp.status_code} @ {b['url']}: {resp.text[:200]}"
                    )
            except (requests.exceptions.RequestException, ConnectionError) as e:
                # 网络类异常：可重试
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_base * (2**attempt))
                    continue
                raise e
            finally:
                if need_acquire and acquired:
                    b["sem"].release()

        # 正常不会到这里
        raise RuntimeError("unexpected retry fallthrough")

    # ----------------------------- 上下文摘要（去 PII，可开关） -----------------------------
    def _build_context_summary(
        self,
        sample: Dict[str, Any],
        report_title: str = "",
        focus_field: str = "",
        limit: int = 500,
    ) -> str:
        alias2canon, _, _ = get_key_alias_maps()
        pii = {
            "姓名",
            "身份证",
            "身份证号",
            "联系方式",
            "联系电话",
            "手机号",
            "电话",
            "住址",
            "地址",
            "联系人",
            "住院号",
            "门诊号",
            "病历号",
        }

        def _is_pii(k: str) -> bool:
            return any(h in k for h in pii)

        def _val_by_alias(k: str):
            v = sample.get(k, "")
            if isinstance(v, str) and v.strip():
                return v.strip()
            ck = alias2canon.get(k, k)
            for rk in sample.keys():
                if rk in {"report", "report_title", "meta"} or _is_pii(rk):
                    continue
                if alias2canon.get(rk, rk) == ck:
                    vv = sample.get(rk, "")
                    if isinstance(vv, str) and vv.strip():
                        return vv.strip()
            return None

        lines: List[str] = []
        seen: set = set()
        title = str(report_title or sample.get("report_title", "") or "").strip()
        keys_map = REPORT_STRUCTURE_MAP.get(title)
        if keys_map:
            f = focus_field
            pos = -1
            for i, n in enumerate(keys_map):
                if n == f or (f and (n in f or f in n)):
                    pos = i
                    break
            if pos != -1:
                for k in [
                    x
                    for x in keys_map[max(0, pos - 2) : min(len(keys_map), pos + 3)]
                    if x != focus_field
                ]:
                    if k not in seen:
                        v = _val_by_alias(k)
                        if isinstance(v, str) and v.strip():
                            lines.append(f"{k}:{v.strip()}")
                            seen.add(k)
        ctx = "\n".join(lines)[:limit]
        return ctx + ("…" if len(ctx) >= limit else "")

    # ----------------------------- RAG 支持 -----------------------------
    def _build_rag_query(
        self, report_title: str, focus_field: str = "", keys: Optional[List[str]] = None
    ) -> str:
        alias2canon, canon2aliases, _ = get_key_alias_maps()
        parts = [s for s in [report_title, focus_field] if s]
        for k in keys or []:
            parts.append(k)
            c = alias2canon.get(k)
            if c:
                parts.extend(canon2aliases.get(c, []))
        return " ".join(parts)

    def _rag_examples(
        self,
        report_title: str,
        focus_field: str = "",
        keys: Optional[List[str]] = None,
        topk: Optional[int] = None,
    ) -> List[str]:
        if not (self.rag_index and self.rag_index.ready()):
            if not self._rag_warned:
                print("[RAG] not used (index not ready).")
                self._rag_warned = True
            return []
        query = self._build_rag_query(report_title, focus_field, keys or [])
        hits = self.rag_index.search(query, topk or self.rag_topk)
        k = max(1, int(topk or self.rag_topk))
        if len(hits) > k:
            pool_n = min(len(hits), max(k * 2, k + 1))
            pool = hits[:pool_n]
            random.shuffle(pool)
            hits = pool[:k]
        out = []
        for idx, _ in hits:
            txt = self.rag_index.chunks[idx]
            out.append(txt[:500] + ("…" if len(txt) > 500 else ""))
        if len(out) > 1:
            random.shuffle(out)
        return out

    # ----------------------------- Prompt 构造与调用（与你的现有逻辑一致） -----------------------------
    def _build_field_prompt(
        self,
        report_title: str,
        field_name: str,
        value: Any,
        context: str = "",
        sample: Optional[Dict[str, Any]] = None,
    ) -> str:
        v = value if isinstance(v := value, str) else json.dumps(v, ensure_ascii=False)
        keys = [
            k
            for k in (sample or {}).keys()
            if isinstance(k, str) and k not in {"report", "report_title", "meta"}
        ]
        if keys:
            random.shuffle(keys)
            keys = keys[: max(6, min(12, len(keys)))]
        rag_refs = self._rag_examples(report_title, field_name + str(value), keys)
        rename_rule = f"field_name 必须等于原字段名：{field_name}"
        prompt = (
            "【任务】根据【参考样本】对下述字段的value做多样化改写（数据增强），风格贴近肿瘤临床病历。\n"
            f"【报告类型】{report_title}\n"
            f"【字段名】{field_name}\n"
            f"【上下文摘要】{(context or '无')}\n"
            "【允许变化】同/近义替换；合理幅度数值或单位微调；语序调整；措辞变化；必要的模板化套语；去掉不属于该字段的描述。\n"
            "【排版要求】禁止 Markdown/围栏。\n"
            + (
                "【参考样本】\n"
                + "\n".join(f"样本{i+1}: {s}" for i, s in enumerate(rag_refs))
                + "\n"
                if rag_refs
                else ""
            )
            + "【输出格式（必须严格遵守）】\n"
            "1) 仅输出一行 JSON（UTF-8）。\n"
            "2) JSON 对象只包含且必须包含两个键，且按此顺序：field_name, value。\n"
            f"3) {rename_rule}\n"
            "4) value 必须是非空字符串；不得为 null/数组/对象。\n"
            "5) 不得新增其他键。\n"
            "【正确示例】\n"
            f'{{"field_name":"{field_name}","value":"用该 field_name在病历上的具体值或具体表达，不要解释和赘述"}}\n'
            "【原值】" + str(v)
        )
        return prompt

    def _augment_one(
        self,
        report_title: str,
        field_name: str,
        value: Any,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        ctx = (
            self._build_context_summary(sample or {}, report_title, field_name)
            if (sample and self.include_context_summary)
            else ""
        )
        data = self._post_report(
            {
                "report": self._build_field_prompt(
                    report_title, field_name, value, ctx, sample
                )
            }
        )
        payload = data.get("report", data)
        if isinstance(payload, str):
            obj = json.loads(payload.replace("```json", "").replace("```", "").strip())
        elif isinstance(payload, dict) and isinstance(payload.get("llm_ret"), str):
            obj = json.loads(
                payload["llm_ret"].replace("```json", "").replace("```", "").strip()
            )
        elif isinstance(payload, dict):
            obj = payload
        else:
            raise ValueError("后端返回格式不正确：期待 JSON 字符串或对象")
        if isinstance(obj, dict) and "field_name" in obj and "value" in obj:
            return str(obj["field_name"]), obj["value"]
        if isinstance(obj, dict) and len(obj) == 1:
            k, v = next(iter(obj.items()))
            return str(k), v
        raise ValueError(f"后端返回缺少必要键(field_name,value)：{_truncate(obj)}")

    def augment_fields(
        self, sample: Dict[str, Any], max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        import concurrent.futures

        keys = [
            k
            for k in sample.keys()
            if k not in {"report", "report_title", "meta"}
            and isinstance(k, str)
            and k.strip()
        ]
        out: Dict[str, Any] = {"report_title": sample.get("report_title", "")}
        if "meta" in sample:
            out["meta"] = sample["meta"]
        workers = max(1, min(len(keys), (max_workers or self.max_workers)))
        title = str(sample.get("report_title", "")).strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(self._augment_one, title, k, sample.get(k), sample): k
                for k in keys
            }
            results: Dict[str, Tuple[str, Any]] = {}
            for fut in concurrent.futures.as_completed(futs):
                orig = futs[fut]
                nk, nv = fut.result()
                results[orig] = (orig, nv)

        for k in keys:
            tk, v = results.get(k, (k, sample.get(k)))
            if tk in out and tk != k:
                out[f"{tk}(alias)"] = v
                out[k] = sample.get(k)
            else:
                out[tk] = v
        return out

    # ----------------------------- 整条合成 -----------------------------
    def _build_record_prompt(self, report_title: str, keys: List[str]) -> str:
        rag_refs = self._rag_examples(report_title, "", keys)
        return (
            "【任务】基于给定报告类型与模块清单，生成完整病历/检查记录（贴近临床）。\n"
            f"【报告类型】{report_title}\n【模块清单】{', '.join(keys)}\n"
            "【排版】不要 Markdown。所有字符串必须用双引号，内部引号需转义，不得出现未闭合的引号。\n"
            + (
                "【参考样本】\n"
                + "\n".join(f"样本{i+1}: {s}" for i, s in enumerate(rag_refs))
                + "\n"
                if rag_refs
                else ""
            )
            + "【输出】仅 JSON 对象；键固定：'report_title' + 模块清单各键；不得新增/遗漏；各 value 为字符串；禁止输出除 JSON 以外的任何内容。"
        )

    def synthesize_record(self, report_title: str, keys: List[str]) -> Dict[str, Any]:
        data = self._post_report(
            {"report": self._build_record_prompt(report_title, keys)}
        )
        payload = data.get("report", data)
        txt = None
        obj = None
        if isinstance(payload, str):
            txt = payload
        elif isinstance(payload, dict) and isinstance(payload.get("llm_ret"), str):
            txt = payload["llm_ret"]
        elif isinstance(payload, dict):
            obj = payload
        else:
            raise ValueError("后端返回格式不正确（整条合成）")

        if obj is None:
            s = (txt or "").replace("```json", "").replace("```", "").strip()
            try:
                obj = json.loads(s)
            except Exception:
                i, j = s.find("{"), s.rfind("}")
                if i != -1 and j != -1 and j > i:
                    s2 = s[i : j + 1]
                    try:
                        obj = json.loads(s2)
                    except Exception:
                        s3 = s2.strip().rstrip(",; ")
                        obj = json.loads(s3)
                else:
                    raise
        if not isinstance(obj, dict):
            raise ValueError("后端返回非 JSON 对象（整条合成）")
        # 放宽约束：report_title 缺失时回退为传入的标题；缺失字段交由上层补救
        title_val = obj.get("report_title", report_title)
        out = {"report_title": title_val}
        for k in keys:
            out[k] = obj.get(k, "")
        return out
