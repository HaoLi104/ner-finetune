# -*- coding: utf-8 -*-
from __future__ import annotations

"""
仅保留 TASK="clean" 逻辑（加上记录级并发 + 中 JSONL 增量落盘）：
- 读取 JSON（列表[dict]，含 report_title / 各字段 / 可选 meta / 可选 report）
- 外层：记录级并发；内层：每条记录内部对字段并发清洗
- 每条记录处理完立即向一个中间 .jsonl 追加一行（包含 __idx 保序）
- 全部完成后把 JSONL 转换为最终 CLEAN_OUT_PATH（JSON 数组，按 __idx 排序，并移除元字段）
- 清洗规则：与字段强相关、去跨字段、能具体则具体、去换行/多余空白、参照 RAG 做"表达对齐但不改结论/数值"
"""

import json
import difflib
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

sys.path.append(".")
from conf import API_KEY
from pre_struct.data_aug.llm_client import OpenAIFieldWiseLLM  # type: ignore

# 可选的 tqdm 进度条
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


# 需忽略的键（不参与清洗/比对）
IGNORED_KEYS = {"report", "report_title", "report_titles", "meta", "added_keys"}


# ----------------------------- RAG 工具：把缓存目录解析为 source_path -----------------------------
def _resolve_rag_source_path(rag_path: Optional[str]) -> Optional[str]:
    """
    - 若 rag_path 为文件：直接返回
    - 若 rag_path 为目录：在目录中查找 rag_*.meta.json/ *.meta.json，读取其中的 source_path 作为真正的 rag_path
    - 若失败：返回 None（表示禁用 RAG）
    """
    if not rag_path:
        return None
    p = Path(rag_path)
    if p.is_file():
        print(f"[RAG] use corpus file: {p}")
        return str(p)
    if p.is_dir():
        metas = sorted(
            list(p.glob("rag_*.meta.json")) + list(p.glob("*.meta.json")),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not metas:
            print(f"[RAG] cache dir has no meta files: {p}")
            return None
        for mf in metas:
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
                src = (
                    meta.get("source_path")
                    or meta.get("path")
                    or meta.get("corpus_path")
                )
                if src:
                    print(f"[RAG] resolved source_path from cache: {src}")
                    return str(src)
            except Exception as e:
                print(f"[RAG] read meta failed: {mf} -> {e}")
        print(f"[RAG] no valid meta found in: {p}")
        return None
    print(f"[RAG] path not exists: {p}")
    return None


def _build_prompt_with_report_title(report_title, field_name, value, keys):
    """
    按报告类型补充更具体的清洗指引，仅作为提示块拼到总 Prompt 中。
    保持"不改医学事实/数值"的底线：仅做表述规范化、单位/格式统一、去冗余。
    """
    title = str(report_title or "").strip()
    fn = str(field_name or "").strip()

    # 心电图（字段多为定量/定性混合）
    if title == "心电图":
        return (
            "心电图字段清洗指引：\n"
            "- RV5、SVI、RV5+SVI：给出单独数值与单位mV，如2.82mV；\n"
            "- P-R间期、QRS时限、QT/QTc：统一单位ms，仅保留本次测量值（QT/QTc可形如410/425ms）；\n"
            "- 电轴：以‘xx°’表示，如’+60°’；\n"
            "- 心律：如’窦性心律/房颤/室早等’，保持定性结论，不扩写；\n"
        )

    # 实验室五大类（典型异常：参考范围拼接、方法学/设备、箭头↑↓、多值/范围）
    if title in [
        "血常规",
        "血生化",
        "凝血功能",
        "肿瘤标志物",
        "甲状腺功能",
        "其他化验单",
    ]:
        return (
            "化验字段清洗指引：\n"
            "- 仅保留本次‘实测值+单位’，丢弃参考范围/方法学/设备名/批号；\n"
            "- 若出现‘数值 + 参考范围(或x-y)’等，保留实测值，去掉参考范围；\n"
            "- 若同时存在‘↑/↓/±/阳性/阴性/1+/2+/3+’，保留该符号或级别；\n"
            "- 单位统一常见形式：g/L, mg/L, mmol/L, U/L, IU/L, %, ng/mL 等，不改变数量级；\n"
        )

    return ""


# ----------------------------- Prompt 构造（含 RAG 表达对齐） -----------------------------
def _build_llm_check_prompt(
    report_title: str,
    field_name: str,
    value: Any,
    context: str = "",
    rag_refs: Optional[List[str]] = None,
    keys: Optional[List[str]] = None,
) -> str:
    v = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    rag_block = ""
    if field_name == "报告医生":
        print()
    if rag_refs:
        rag_block = (
            f"【RAG样例（用于对齐语言表达，从中获取{field_name}的表达，且要尽量保持表达的多样性,如果下面样例中没有{field_name}的表达，就无需参考）】:\n"
            + "\n".join(f"样例{i+1}: {s}" for i, s in enumerate(rag_refs))
            + "\n"
        )
    other_keys = [k for k in keys if k != field_name]
    # if field_name == "患者信息":
    # print(field_name,other_keys)
    return (
        f"【任务】改写并清洗{field_name}的值,达到数据增强的目的，数据要足够干净（删除与{field_name}无关的内容），生成的数据是用于训练模型根据字段名识别value值。\n"
        f"要求：参考【RAG样例】中真实出现的医学表达、术语风格，对原值进行语言风格统一、内容清洗修正和多样化改写，使其更贴近RAG样例中的专业表达，若原值中存在其他字段的内容，请删除。\n"
        "如果原值的表达已经足够专业，不要对原值大幅度改写，只需要删除其他字段的内容。"
        f"1) 仅保留与“{field_name}”字段*密切相关*的信息,不得包含其他字段的内容,严格删除与{', '.join(other_keys)}等其他字段相关的内容；\n"
        "2) 若原值中包含不具体的人名、住址、单位等表述，应仿照RAG样例改写为合规的虚拟示例，不得用真实姓名；\n"
        "3) 去除不必要的换行符；\n"
        "4) 可将原始医学结论、术语、数值单位等，参考RAG样例进行专业化表达改写，但禁止随意杜撰；\n"
        "5) 若原值为空或为“见上/如下/详见报告/无”等不具体表达，可结合RAG样例给出一个合理、合规的具体示例值（除非该字段确实允许“无/未见异常”作为结论）；\n"
        f"【报告类型】{report_title}\n{_build_prompt_with_report_title(report_title, field_name, value, keys)}"
        f"【字段名】{field_name}\n"
        + rag_block
        + "【输出】不要Markdown，仅输出JSON对象。\n"
        f'【输出示例】{{"field_name":"{field_name}","value":"只包含{field_name}的值，请勿输出key，直接输出value，严禁出现其他字段的内容。避免出现值后面再黏连其他内容，误导后期模型训练"}}\n'
        "【原值】" + str(v)
    )
def _llm_check_one(
    llm: OpenAIFieldWiseLLM,
    report_title: str,
    field_name: str,
    value: Any,
    sample: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    # 上下文摘要
    try:
        ctx = llm._build_context_summary(sample or {}, report_title, field_name)  # type: ignore[attr-defined]
    except Exception:
        ctx = ""

    # RAG 参考样本
    keys = []
    if isinstance(sample, dict):
        keys = [
            k
            for k in sample.keys()
            if isinstance(k, str) and k not in {"report", "report_title", "meta"}
        ]
    try:
        rag_refs = llm._rag_examples("", field_name + ":"+value, [])  # type: ignore[attr-defined]
    except Exception:
        rag_refs = []

    prompt = _build_llm_check_prompt(
        report_title, field_name, value, ctx, rag_refs, keys
    )
    data = llm._post_report({"report": prompt})  # type: ignore[attr-defined]

    payload = data.get("report", data) if isinstance(data, dict) else data
    try:
        if isinstance(payload, str):
            s = payload.replace("```json", "").replace("```", "").strip()
            obj = json.loads(s)
        elif isinstance(payload, dict) and isinstance(payload.get("llm_ret"), str):
            s = payload["llm_ret"].replace("```json", "").replace("```", "").strip()
            obj = json.loads(s)
        elif isinstance(payload, dict):
            obj = payload
        else:
            obj = {}
    except Exception:
        obj = {}

    # 解析结果与兜底
    try:
        fn = str(obj.get("field_name", field_name))
        val = obj.get("value", value)
        val = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    except Exception:
        fn = field_name
        val = value if isinstance(value, str) else str(value)

    # 去除多余空白
    val = re.sub(r"\s+", " ", str(val)).strip()
    return fn, val


# ----------------------------- 退化打印进度（无 tqdm 时） -----------------------------
def _print_progress(i_done: int, total: int, t0: float, step: int = 25) -> None:
    """每 step 条打印一次：百分比 + 速率 + ETA"""
    if i_done % step != 0 and i_done != total:
        return
    elapsed = max(1e-6, time.monotonic() - t0)
    rate = i_done / elapsed  # rec/s
    remain = (total - i_done) / rate if rate > 0 else 0.0
    pct = (i_done / total) * 100 if total else 100.0
    print(
        f"[llm_clean] {i_done}/{total} ({pct:5.1f}%) | {rate:5.2f} rec/s | ETA {remain:5.1f}s",
        flush=True,
    )


# ----------------------------- 单条记录处理（含字段级并发） -----------------------------
def _process_one_record(
    idx: int,
    rec: Dict[str, Any],
    llm: OpenAIFieldWiseLLM,
    inner_workers: int,
    target_report_types: Optional[List[str]] = None,
    process_mode: str = "added_keys_only",
    custom_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
    max_fields_per_record: Optional[int] = None,
) -> Dict[str, Any]:
    import concurrent.futures

    title = str(rec.get("report_title", "")).strip()

    # 检查是否处理该报告类型
    if target_report_types is not None and title not in target_report_types:
        print(f"[llm_clean] 跳过未指定的报告类型：{title}")
        return rec

    # 根据处理模式选择要处理的字段
    if process_mode == "added_keys_only":
        # 只处理 added_keys 中的字段
        added_keys = rec.get("added_keys", [])
        if added_keys:
            keys = [k for k in added_keys if isinstance(k, str) and k.strip()]
        else:
            keys = []  # 没有added_keys则跳过
    elif process_mode == "all_fields":
        # 处理所有非忽略字段
        keys = [
            k
            for k in rec.keys()
            if isinstance(k, str) and k.strip() and k not in IGNORED_KEYS
        ]
    elif process_mode == "custom_fields":
        # 处理指定的自定义字段列表
        if custom_fields:
            keys = [
                k
                for k in custom_fields
                if k in rec and isinstance(k, str) and k.strip()
            ]
        else:
            keys = []
    else:
        keys = []

    # 应用排除字段过滤
    if exclude_fields:
        keys = [k for k in keys if k not in exclude_fields]

    # 限制字段数量
    if max_fields_per_record and len(keys) > max_fields_per_record:
        keys = keys[:max_fields_per_record]

    new_rec: Dict[str, Any] = {"__idx": idx, "report_title": title}
    if "meta" in rec:
        new_rec["meta"] = rec["meta"]

    workers = max(1, min(len(keys), int(inner_workers)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_llm_check_one, llm, title, k, rec.get(k, ""), rec): k
            for k in keys
        }
        for fut in concurrent.futures.as_completed(futs):
            k = futs[fut]
            try:
                _, val = fut.result()
            except Exception as e:  # 字段失败回退原值并标记
                new_rec.setdefault("__field_errors", {})[k] = str(e)
                val = rec.get(k, "")
            new_rec[k] = val if isinstance(val, str) else str(val)

    # 保留所有其他字段（不在IGNORED_KEYS中的字段）
    for k, v in rec.items():
        # 跳过忽略的字段和已经处理的字段
        if k in IGNORED_KEYS or k in new_rec:
            continue
        new_rec[k] = v

    # 如果有added_keys，也需要保留它
    if "added_keys" in rec:
        new_rec["added_keys"] = rec["added_keys"]

    if "report" in rec:
        new_rec["report"] = rec["report"]
    return new_rec


# ----------------------------- JSONL 转 JSON 数组 -----------------------------
def _jsonl_to_json_array(jsonl_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return items
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                continue
    # 按 __idx 排序并去掉内部元字段
    items.sort(key=lambda x: x.get("__idx", 0))
    for it in items:
        it.pop("__idx", None)
        it.pop("__field_errors", None)
    return items


# ----------------------------- 主流程：记录级并发 + JSONL 增量写 -----------------------------
def llm_clean_fields_only(
    in_path: str,
    out_path: str,
    base_url: str = "https://qwen3.yoo.la/v1/",
    api_key: Optional[str] = None,
    model: str = "qwen3-32b",
    record_workers: int = 6,  # 外层并发：同时处理多少条记录
    inner_workers: int = 12,  # 内层并发：单条记录内并发字段数
    rag_path: Optional[str] = None,
    rag_topk: int = 2,
    changes_jsonl_path: Optional[str] = None,
    target_report_types: Optional[
        List[str]
    ] = None,  # 目标报告类型列表，None表示处理所有类型
    # 字段处理范围控制
    process_mode: str = "added_keys_only",  # "added_keys_only", "all_fields", "custom_fields"
    custom_fields: Optional[
        List[str]
    ] = None,  # 当process_mode="custom_fields"时指定要处理的字段
    exclude_fields: Optional[List[str]] = None,  # 要排除的字段列表
    max_fields_per_record: Optional[int] = None,  # 每条记录最多处理多少个字段
) -> None:
    # 读入数据
    data: List[Dict[str, Any]] = json.loads(Path(in_path).read_text(encoding="utf-8"))

    # 按报告类型过滤（数量控制已在上游完成）
    if target_report_types is not None:
        filtered_data = []
        type_counts = {}

        for rec in data:
            title = str(rec.get("report_title", "")).strip()

            # 检查报告类型
            if target_report_types is not None and title not in target_report_types:
                continue

            # 统计类型数量（仅用于显示）
            type_counts[title] = type_counts.get(title, 0) + 1
            filtered_data.append(rec)

        data = filtered_data
        print(f"[filter] after filtering by report types {target_report_types}")
        for title, count in type_counts.items():
            print(f"  {title}: {count} samples")

    n = len(data)
    print(f"[start] total records after filtering: {n}")

    # RAG 解析
    resolved_rag = _resolve_rag_source_path(rag_path) if rag_path else None
    if rag_path and not resolved_rag:
        print("[RAG] disabled (cannot resolve source_path from cache dir).")
    elif resolved_rag:
        print(f"[RAG] using source_path: {resolved_rag}")

    # LLM 客户端（线程安全；内部已做后端负载控制）
    llm = OpenAIFieldWiseLLM(
        model=model,
        base_url=[s.strip() for s in str(base_url).split("|") if s.strip()],
        api_key=api_key,
        max_workers=inner_workers,
        rag_path=resolved_rag,
        rag_topk=rag_topk,
    )

    # 变更记录 JSONL 路径（仅记录有变更的字段）
    changes_path = Path(changes_jsonl_path or (str(out_path) + ".changes.jsonl"))
    changes_path.parent.mkdir(parents=True, exist_ok=True)
    if changes_path.exists():
        print(f"[warn] remove existing changes JSONL: {changes_path}")
        changes_path.unlink()

    # 线程安全文件写锁
    file_lock = threading.Lock()
    processed_data: List[Dict[str, Any]] = []  # 存储处理结果

    import concurrent.futures

    t0 = time.monotonic()
    bar = tqdm(total=n, desc="[clean] records", ncols=100) if tqdm else None
    done = 0

    # 外层并发：提交所有记录
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(record_workers)) as ex:
        futs = {
            ex.submit(
                _process_one_record,
                i,
                rec,
                llm,
                inner_workers,
                target_report_types,
                process_mode,
                custom_fields,
                exclude_fields,
                max_fields_per_record,
            ): i
            for i, rec in enumerate(data)
        }
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                new_rec = fut.result()
            except Exception as e:
                # 记录失败也写入 JSONL，保留原始字段（尽量不丢行）
                err_rec = {"__idx": i, "__error": str(e)}
                err_rec.update(data[i])
                new_rec = err_rec

            # 处理完成的记录直接添加到结果列表
            with file_lock:
                processed_data.append(new_rec)

            # 如果与原始值不同，记录变更到单独的 changes JSONL，并在日志仅打印"被删除的内容"
            try:
                orig_rec = data[i]
                title = str(orig_rec.get("report_title", "")).strip()
                # 比较所有字段，而不仅仅是 added_keys 中的字段或非忽略字段
                keys = [k for k in orig_rec.keys() if isinstance(k, str) and k.strip()]
                changes_lines: List[str] = []
                log_lines: List[str] = []
                for k in keys:
                    before_val = orig_rec.get(k, "")
                    after_val = new_rec.get(k, "")
                    before_str = (
                        before_val if isinstance(before_val, str) else str(before_val)
                    )
                    after_str = (
                        after_val if isinstance(after_val, str) else str(after_val)
                    )
                    if before_str != after_str:
                        change_obj = {
                            "__idx": i,
                            "report_title": title,
                            "field": k,
                            "before": before_val,
                            "after": after_val,
                        }
                        changes_lines.append(json.dumps(change_obj, ensure_ascii=False))
                        # 打印处理前后的完整内容对比
                        try:
                            # 清理和截断before/after内容以便显示
                            before_clean = re.sub(r"\s+", " ", before_str).strip()
                            after_clean = re.sub(r"\s+", " ", after_str).strip()
                            
                            # 设置最大显示长度，避免日志过长
                            max_len = 150
                            if len(before_clean) > max_len:
                                before_display = before_clean[:max_len] + "…"
                            else:
                                before_display = before_clean
                                
                            if len(after_clean) > max_len:
                                after_display = after_clean[:max_len] + "…"
                            else:
                                after_display = after_clean
                            
                            # 构建前后对比日志
                            log_lines.append(
                                f"[change] idx={i} title={title} field={k}\n"
                                f"  BEFORE: {before_display}\n"
                                f"  AFTER:  {after_display}"
                            )
                        except Exception:
                            # 回退：若处理失败，使用简单格式
                            log_lines.append(
                                f"[change] idx={i} title={title} field={k} | content changed"
                            )
                if changes_lines:
                    with file_lock:
                        with changes_path.open("a", encoding="utf-8") as cf:
                            for l in changes_lines:
                                cf.write(l + "\n")
                        # 同步打印日志
                        for msg in log_lines:
                            print(msg, flush=True)
            except Exception:
                pass

            done += 1
            if bar:
                bar.update(1)
            else:
                _print_progress(done, n, t0, step=max(10, n // 40))

    if bar:
        bar.close()

    # 直接使用处理结果并落盘
    final_items = processed_data
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(final_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.monotonic() - t0
    print(
        f"[ok] llm_clean -> {out_path} | records={len(final_items)}/{n} | elapsed: {elapsed:.1f}s"
    )
    print(f"[diff] changes jsonl kept at: {changes_path}")


# ----------------------------- 仅 clean -----------------------------
if __name__ == "__main__":
    import os
    from datetime import datetime

    # 配置参数
    TARGET_REPORT_TYPES = None # 指定要处理的报告类型，None表示处理所有类型
    # MAX_SAMPLES_PER_TYPE 不需要在这里指定，数量控制在上游 run_da.py 中已完成

    # 输入/输出 - v5.0 第二阶段：字段清洗
    DATE_TAG = os.environ.get("DA_DATE_TAG") or datetime.now().strftime("%Y%m%d")
    BASE_DIR = Path("data") / DATE_TAG
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 允许外部覆盖输入/输出
    CLEAN_IN_PATH = os.environ.get("DA_CLEAN_IN") or str(BASE_DIR / "clean_ocr_ppt_da_v5_0_field_added.json")
    CLEAN_OUT_PATH = os.environ.get("DA_CLEAN_OUT") or str(BASE_DIR / "clean_ocr_ppt_da_v5_0_field_cleaned.json")
    CHANGES_JSONL_PATH = os.environ.get("DA_CHANGES_OUT") or str(BASE_DIR / "clean_ocr_ppt_da_v5_0_field_cleaned.changes.jsonl")

    # 后端与并发 - 优化配置（32核CPU，内存充足）
    BASE_URL = "https://qwen3.yoo.la/v1/|http://123.57.234.67:8000/v1"
    MODEL = "qwen3-32b"
    RECORD_WORKERS = 6  # 外层并发度（记录级，充分利用CPU）
    INNER_WORKERS = 6   # 内层并发度（字段级，平衡吞吐与稳定性）

    # RAG
    RAG_PATH = "../data/rag/ocr_summary_words.txt"
    RAG_TOPK = 3

    # 字段处理范围控制
    PROCESS_MODE = "added_keys_only"  # "added_keys_only", "all_fields", "custom_fields"
    CUSTOM_FIELDS = None  # 自定义字段列表，当PROCESS_MODE="custom_fields"时使用
    EXCLUDE_FIELDS = None  # 要排除的字段列表
    MAX_FIELDS_PER_RECORD = 10  # 每条记录最多处理多少个字段

    llm_clean_fields_only(
        in_path=CLEAN_IN_PATH,
        out_path=CLEAN_OUT_PATH,
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        record_workers=RECORD_WORKERS,
        inner_workers=INNER_WORKERS,
        rag_path=RAG_PATH,
        rag_topk=RAG_TOPK,
        changes_jsonl_path=CHANGES_JSONL_PATH,
        target_report_types=TARGET_REPORT_TYPES,
        # 字段处理范围控制
        process_mode=PROCESS_MODE,
        custom_fields=CUSTOM_FIELDS,
        exclude_fields=EXCLUDE_FIELDS,
        max_fields_per_record=MAX_FIELDS_PER_RECORD,
    )
