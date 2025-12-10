#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_qwen32b: LLM-based key-value evaluation

Workflow
- Read eval data: data/kv_ner_prepared/val_eval.jsonl
- For each sample, call utils.call_llm.call_llm with a constrained prompt
- Save raw LLM outputs to this directory as llm_outputs.jsonl
- Convert LLM outputs to semi-structured entities and evaluate with evaluation library
- Save summary to runs/experiments/compare_qwen32b/eval/eval_summary.json

Notes
- Metrics reported: text-exact and text-overlap (position is optional/naive)
- Keys are restricted to the gold keys of each sample for fair comparison
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
sys.path.append(".")
from utils.call_llm import call_llm, call_llm_many  # type: ignore
from evaluation.src.easy_eval import evaluate_entities  # type: ignore

# progress bar (with safe fallback)
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


@dataclass
class Args:
    input_file: str
    model: str
    out_dir: str
    pred_out: str
    limit: int
    report_types: str
    exclude_keys: str
    concurrency: int
    rate_limit: float
    retries: int


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                data.append(json.loads(ln))
    return data


def _strip_trailing_punct(text: str, start: int, end: int) -> Tuple[str, int, int]:
    trailing = set(' \t\u3000\r\n。，、；：？！,.;:?![]【】()（）「」『』""\'\'…—')
    orig = text
    while text and text[0] in trailing:
        text = text[1:]
        start += 1
    while text and text[-1] in trailing:
        text = text[:-1]
        end -= 1
    if not text:
        return orig, start, end
    return text, start, end


def _normalize_text_for_eval(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("\u3000", " ")
    s = re.sub(r"^\s+|\s+$", "", s)
    edge_punct = "。，、；:;,:()[]{}<>"
    i = 0
    while i < len(s) and s[i] in edge_punct:
        i += 1
    j = len(s)
    while j > i and s[j - 1] in edge_punct:
        j -= 1
    return s[i:j]


def _to_eval_pack(kv_map: Dict[str, Any], *, exclude: List[str] | None = None) -> Dict[str, Any]:
    exclude = exclude or []
    ents: List[Dict[str, Any]] = []
    for k, v in (kv_map or {}).items():
        if k in exclude:
            continue
        if not isinstance(v, dict):
            continue
        try:
            s = int(v.get("start", -1))
            e = int(v.get("end", -1))
            t = str(v.get("text", "")).strip()
        except Exception:
            continue
        if s >= 0 and e > s and t:
            t_clean, s2, e2 = _strip_trailing_punct(t, s, e)
            ents.append({"start": s2, "end": e2, "text": _normalize_text_for_eval(t_clean)})
    return {"entities": ents}


def _pred_pack_from_strings(pred_map: Dict[str, str], ref_keys: List[str], *, exclude: List[str] | None = None, base_text: str | None = None) -> Dict[str, Any]:
    exclude = exclude or []
    ents: List[Dict[str, Any]] = []
    for k in ref_keys:
        if k in exclude:
            continue
        v = pred_map.get(k)
        if not isinstance(v, str):
            continue
        t = _normalize_text_for_eval(v.strip())
        if not t:
            continue
        # Try to locate in base_text for optional position metric compatibility
        if base_text:
            pos = base_text.find(v.strip())
            if pos >= 0:
                ents.append({"start": pos, "end": pos + len(v.strip()), "text": t})
                continue
        # Fallback: positions are not required for text-based matching
        ents.append({"start": 0, "end": max(1, len(t)), "text": t})
    return {"entities": ents}


def _extract_json_object(s: str) -> Dict[str, Any] | None:
    """Try to extract a JSON object from the string.
    Handles plain JSON or fenced code blocks. Returns dict or None.
    """
    s = s.strip()
    # direct load
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # code fences
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", s, flags=re.S | re.I)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # first balanced braces (simple, not string-safe but works often)
    try:
        start = s.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        obj = json.loads(s[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                        break
    except Exception:
        pass
    return None


def _load_llm_prompt_template() -> str:
    """Load the full multi-example prompt from utils/call_llm.py.

    Returns the exact triple-quoted template (containing 示例1/2/3 和
    "输入文本:{ocr_text}" 占位符)。如果提取失败，回退到简短模板。
    """
    try:
        # Resolve repo root then utils/call_llm.py
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[3]
        call_llm_path = repo_root / "utils" / "call_llm.py"
        txt = call_llm_path.read_text(encoding="utf-8")
        m = re.search(r"prompt\s*=\s*\"\"\"([\s\S]*?)\"\"\"\.replace\(", txt)
        if m:
            return m.group(1)
    except Exception:
        pass
    # Fallback minimal template
    return (
        "我正在进行病历半结构化任务，请根据以下报告内容，提取报告中所有的信息,最终以key:value的形式输出。\n"
        "输入文本:{ocr_text}\n"
        "请只返回提取到的信息，不要添加多余内容，也不要推理，直接截取所有的信息。\n"
        "输出格式,按照key、value在输入文本中的出现顺序输出:{{\"key\":\"value\"}}\n"
    )


def build_prompt(report_title: str, report_text: str, keys: List[str]) -> str:
    # 直接复用 utils/call_llm.py 中包含三个示例的大模板，并替换占位符
    tpl = _load_llm_prompt_template()
    keys_part = "，".join(keys)
    insert_text = (
        f"报告类型：{report_title}\n"
        f"键列表：[{keys_part}]\n"
        f"报告全文：\n{report_text}"
    )
    return tpl.replace("{ocr_text}", insert_text)


def run(args: Args) -> None:
    in_path = Path(args.input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    samples = _read_jsonl(in_path)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    pred_out = Path(args.pred_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    # Optional filtering and exclusions
    report_type_filter: List[str] = [s for s in (args.report_types.split(",") if args.report_types else []) if s]
    exclude_keys: List[str] = [s for s in (args.exclude_keys.split(",") if args.exclude_keys else []) if s]

    # Prepare jobs
    jobs = []
    for item in samples:
        report_title = str(item.get("report_title", "") or "")
        report_text = str(item.get("report", "") or "").replace("\r\n", "\n").replace("\r", "\n")
        if not report_text.strip():
            continue
        if report_type_filter and report_title not in report_type_filter:
            continue
        gold_map: Dict[str, Any] = item.get("spans", {}) or {}
        ref_keys = list(gold_map.keys())
        prompt = build_prompt(report_title, report_text, ref_keys)
        jobs.append({
            "report_index": item.get("report_index"),
            "report_title": report_title,
            "report_text": report_text,
            "gold_map": gold_map,
            "ref_keys": ref_keys,
            "prompt": prompt,
        })

    # Concurrent LLM calls
    prompts = [j["prompt"] for j in jobs]
    raw_outputs = call_llm_many(
        prompts,
        model=args.model,
        max_workers=max(1, args.concurrency),
        retries=max(0, args.retries),
        rate_limit_per_sec=(args.rate_limit if args.rate_limit > 0 else None),
        show_progress=True,
    )

    # Collect evaluation packs
    y_true: List[Dict[str, Any]] = []
    y_pred_exact: List[Dict[str, Any]] = []

    with pred_out.open("w", encoding="utf-8") as f_pred:
        for job, raw in zip(jobs, raw_outputs):
            parsed = _extract_json_object(raw)
            pred_map: Dict[str, str] = {}
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if k in job["ref_keys"] and isinstance(v, (str, int, float)):
                        pred_map[k] = str(v)
            f_pred.write(
                json.dumps(
                    {
                        "report_index": job["report_index"],
                        "report_title": job["report_title"],
                        "pred": pred_map,
                        "raw": raw,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            y_true.append(_to_eval_pack(job["gold_map"], exclude=exclude_keys))
            y_pred_exact.append(
                _pred_pack_from_strings(
                    pred_map,
                    job["ref_keys"],
                    exclude=exclude_keys,
                    base_text=job["report_text"],
                )
            )

    # Evaluate text-exact and text-overlap (and naive position)
    res_exact = evaluate_entities(
        y_true,
        y_pred_exact,
        mode="semi_structured",
        matching_method="text",
        text_match_mode="exact",
    )
    res_overlap = evaluate_entities(
        y_true,
        y_pred_exact,
        mode="semi_structured",
        matching_method="text",
        text_match_mode="overlap",
    )
    res_pos = evaluate_entities(
        y_true,
        y_pred_exact,
        mode="semi_structured",
        matching_method="position",
    )

    # Persist summary under runs/experiments
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "text_exact": res_exact,
        "text_overlap": res_overlap,
        "position": res_pos,
        "config_path": f"scripts/experiments/compare_qwen32b/config.json",
        "input_file": str(in_path),
        "model": args.model,
        "num_samples": len(y_true),
        "exclude_keys": exclude_keys,
        "report_types": report_type_filter or None,
        "predictions_path": str(pred_out),
        "concurrency": args.concurrency,
        "rate_limit": args.rate_limit if args.rate_limit > 0 else None,
        "retries": args.retries,
    }
    (out_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Evaluate Qwen-32B extraction vs gold KV spans")
    ap.add_argument("--input", dest="input_file", default="data/kv_ner_prepared/val_eval.jsonl")
    ap.add_argument("--model", dest="model", default="qwen3-32b")
    ap.add_argument(
        "--out_dir",
        default="runs/experiments/compare_qwen32b/eval",
        help="Evaluation summary output dir",
    )
    ap.add_argument(
        "--pred_out",
        default="scripts/experiments/compare_qwen32b/llm_outputs.jsonl",
        help="Where to save raw LLM predictions",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    ap.add_argument("--concurrency", type=int, default=8, help="Parallel workers for LLM calls")
    ap.add_argument("--rate_limit", type=float, default=0.0, help="Max requests per second (0=unlimited)")
    ap.add_argument("--retries", type=int, default=2, help="Retries per request on failure")
    ap.add_argument(
        "--report_types",
        type=str,
        default="",
        help="Comma-separated report titles to include (default: all)",
    )
    ap.add_argument(
        "--exclude_keys",
        type=str,
        default="无键名,其他",
        help="Comma-separated keys to exclude from evaluation",
    )
    ns = ap.parse_args()
    return Args(
        input_file=ns.input_file,
        model=ns.model,
        out_dir=ns.out_dir,
        pred_out=ns.pred_out,
        limit=int(ns.limit or 0),
        report_types=ns.report_types,
        exclude_keys=ns.exclude_keys,
        concurrency=int(ns.concurrency or 1),
        rate_limit=float(ns.rate_limit or 0.0),
        retries=int(ns.retries or 0),
    )


if __name__ == "__main__":
    run(parse_args())
