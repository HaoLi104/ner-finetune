# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
# 添加项目根目录到路径，支持直接运行
if __name__ == "__main__":
    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch  # set_seed 用

# -------------------------
# 依赖导入（带兜底）
# -------------------------
_USE_SW = os.environ.get("EVAL_USE_SLIDE_WINDOW", "0") == "1"

if _USE_SW:
    # 使用 slide_window 的推理与配置
    from pre_struct.slide_window.test_slide_window import (
        load_ebqa as _sw_load_ebqa,
        predict_one,
        PredictConfig,  # 类型占位
    )
    from pre_struct.slide_window.config_io import (
        load_config as load_ebqa_cfg,
        resolve_model_dir,
        resolve_tokenizer_name,
        resolve_report_struct_path,
        lengths_from,
        chunk_mode_from,
        predict_block,
    )
    from pre_struct.ebqa.da_core.dataset import export_test_spans

    # 适配接口：评估逻辑期望 load_ebqa(cfg)->(model,collate,device)
    def load_ebqa(cfg):  # type: ignore
        model, collate, device, _ = _sw_load_ebqa()
        return model, collate, device
else:
    try:
        # 推理侧（EBQA）
        from pre_struct.ebqa.test_ebqa import (
            load_ebqa,
            predict_one,
            PredictConfig,
        )
        # 导出 gold spans（新版签名）
        from pre_struct.ebqa.da_core.dataset import export_test_spans
        from pre_struct.ebqa.config_io import (
            load_config as load_ebqa_cfg,
            resolve_model_dir,
            resolve_tokenizer_name,
            resolve_report_struct_path,
            lengths_from,
            chunk_mode_from,
            predict_block,
        )
    except Exception:
        # 当从 pre_struct/ 目录内运行时的兜底
        from ebqa.test_ebqa import (
            load_ebqa,
            predict_one,
            PredictConfig,
        )
        from ebqa.da_core.dataset import export_test_spans
        from ebqa.config_io import (
            load_config as load_ebqa_cfg,
            resolve_model_dir,
            resolve_tokenizer_name,
            resolve_report_struct_path,
            lengths_from,
            chunk_mode_from,
            predict_block,
        )

# 评测库
sys.path.append(".")
from evaluation.src.easy_eval import evaluate_entities  # type: ignore

# 进度条
try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, *args, **kwargs):
        return x


# =========================
# 小工具
# =========================
def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_content(content: str) -> str:
    """必要时做清理；目前保持原样。"""
    return content


def _read_json_or_jsonl(p: Path) -> list:
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


def _is_valid_span(v: dict) -> bool:
    """有效 span：start>=0, end>start, text 非空。"""
    try:
        s = int(v.get("start", -1))
        e = int(v.get("end", -1))
        t = str(v.get("text", "")).strip()
        return (s >= 0) and (e > s) and bool(t)
    except Exception:
        return False


def strip_trailing_punctuation(text: str, start: int, end: int) -> tuple[str, int, int]:
    """去除开头和末尾的标点符号，并调整start/end位置
    
    Args:
        text: 文本内容
        start: 起始位置（字符级）
        end: 结束位置（字符级，包含）
    
    Returns:
        (清理后的text, 调整后的start, 调整后的end)
    """
    # 定义需要去除的首尾符号
    trailing_puncts = set('。，、；：？！,.;:?![]【】()（）「」『』""\'\'…—')
    
    original_text = text
    
    # 从开头去除标点（特别是冒号）
    while text and text[0] in trailing_puncts:
        text = text[1:]
        start += 1
    
    # 从末尾开始去除标点
    while text and text[-1] in trailing_puncts:
        text = text[:-1]
        end -= 1
    
    # 如果全部被去除了，保留原始值
    if not text:
        return original_text, start, end
    
    return text, start, end


def _to_eval_pack(spans_or_preds: Dict[str, Any], exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    将 {key: {start,end,text}} 打包为:
        {"entities": [{"start": s, "end": e, "text": t}, ...]}
    仅收集"有效 span"。在计算position时会去除末尾的标点符号。
    
    Args:
        spans_or_preds: 键值对字典
        exclude_keys: 要排除的key列表（如["无键名"]）
    """
    exclude_keys = exclude_keys or []
    ents: List[Dict[str, Any]] = []
    
    for k, v in (spans_or_preds or {}).items():
        # 跳过排除的key
        if k in exclude_keys:
            continue
        
        if not isinstance(v, dict):
            continue
        if not _is_valid_span(v):
            continue
        s = int(v["start"])
        e = int(v["end"])
        t = clean_content(str(v.get("text", "")).strip())
        
        # 去除末尾标点符号，并调整位置
        t_clean, s_clean, e_clean = strip_trailing_punctuation(t, s, e)
        
        ents.append({"start": s_clean, "end": e_clean, "text": t_clean})
    return {"entities": ents}


# =========================
# 构造 ground truth 子集（与训练口径一致）
# =========================
def ground_truth(
    test_data_path: str,
    tok,
    out_dir: str = "data",
    max_samples: int = 200,
    max_report_tokens: int = 500,
    seed: Optional[int] = 42,
    struct_path: Optional[str] = "keys/keys_merged.json",
) -> Dict[str, str]:
    """
    从原始 JSON/JSONL 中抽取子集样本，满足：
      - 最多 max_samples 条；
      - 报告正文按 tokenizer 计的 token 数 < max_report_tokens。
    对“子集”导出 gold spans（只按标题键；缺失键写 start=-1,end=-1,text=""）。
    """
    src = Path(test_data_path)
    if not src.exists():
        raise FileNotFoundError(f"test_data_path not found: {test_data_path}")
    all_items = _read_json_or_jsonl(src)
    if not all_items:
        raise RuntimeError(f"Empty test file: {test_data_path}")

    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(all_items)

    picked, out_items = 0, []
    m = max(1, int(max_samples)) if (max_samples and max_samples > 0) else 10**9
    thr = max(1, int(max_report_tokens))

    for it in tqdm(all_items, desc="[subset] scanning reports"):
        if picked >= m:
            break
        rep = str(it.get("report", "") or "").replace("\r\n", "\n").replace("\r", "\n")
        if not rep.strip():
            continue
        try:
            tok_len = len(tok.tokenize(rep))
        except Exception:
            continue
        if tok_len < thr:
            out_items.append(it)
            picked += 1

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    out_json = out_base / "ground_truth.json"
    out_jsonl = out_base / "ground_truth.jsonl"

    out_json.write_text(
        json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 导出 spans（缺失键写 start=-1,end=-1,text=""）
    export_test_spans(
        str(out_json),
        str(out_jsonl),
        show_progress=True,
        report_struct_path=struct_path,
        only_title_keys=True,
    )

    print(f"[OK] subset={len(out_items)} saved -> {out_json} / {out_jsonl}")
    return {"json_path": str(out_json), "jsonl_path": str(out_jsonl)}


# =========================
# 评测主流程
# =========================
def evaluate_model(
    model,
    collate,
    cfg: PredictConfig,
    test_data_path: str,
    align_mode: str = "gold",  # 'gold' | 'pred' | 'union'
    error_dump_path: str = "data/test_semi_struct/error_positions.jsonl",
    report_types_filter: Optional[List[str]] = None,  # 新增：指定报告类型过滤
    error_threshold: float = 0.99,  # 新增：错误样本F1阈值
    exclude_keys: Optional[List[str]] = None,  # 新增：排除的key列表
):
    """
    对 ground_truth.jsonl 做评测（半结构化）：
      - 位置严格匹配（position）
      - 文本严格匹配（text exact）
      - 文本重叠（text overlap）
    仅评测"有效 span"；对齐后两边都为空的样本跳过，不计入平均。
    
    错误样本记录：当position的F1 < error_threshold时，记录该样本到jsonl文件，
    并且只保留位置不匹配的实体（过滤掉位置完全相同的实体）。
    """
    totals_pos: Dict[str, float] = {}
    totals_txt_exact: Dict[str, float] = {}
    totals_txt_overlap: Dict[str, float] = {}
    counted_reports = 0

    err_path = Path(error_dump_path)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    with open(test_data_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    for line in tqdm(lines, desc="[eval] reports"):
        item = json.loads(line)

        gold_raw = item.get("spans", {}) or {}
        report_title = str(item.get("report_title", "") or "")
        report_text = str(item.get("report", "") or "")
        
        # 跳过空报告
        if not report_text.strip():
            continue
        
        # 报告类型过滤（如果指定）
        if report_types_filter:
            if report_title not in report_types_filter:
                continue

        # 预测（遵循推理侧实现）
        preds_full = predict_one(cfg, model, collate, report_title, report_text)

        # 只保留“有效 span”
        gold_valid = {
            k: {
                "start": int(v.get("start", -1)),
                "end": int(v.get("end", -1)),
                "text": clean_content(str(v.get("text", "")).strip()),
            }
            for k, v in gold_raw.items()
            if isinstance(v, dict) and _is_valid_span(v)
        }
        pred_valid = {
            k: {
                "start": int(v.get("start", -1)),
                "end": int(v.get("end", -1)),
                "text": clean_content(str(v.get("text", "")).strip()),
            }
            for k, v in (preds_full or {}).items()
            if isinstance(v, dict) and _is_valid_span(v)
        }

        # 对齐：只对“有效键”对齐
        am = str(align_mode or "gold").lower()
        if am not in {"gold", "pred", "union"}:
            am = "gold"
        if am == "gold":
            ref_keys = set(gold_valid.keys())
        elif am == "pred":
            ref_keys = set(pred_valid.keys())
        else:
            ref_keys = set(gold_valid.keys()) | set(pred_valid.keys())

        true_map = {k: gold_valid[k] for k in ref_keys if k in gold_valid}
        pred_map = {k: pred_valid[k] for k in ref_keys if k in pred_valid}

        data_true = _to_eval_pack(true_map, exclude_keys=exclude_keys)
        data_pred = _to_eval_pack(pred_map, exclude_keys=exclude_keys)

        # 两边都没有实体 -> 跳过该报告
        if not data_true["entities"] and not data_pred["entities"]:
            continue

        # 评测 1：位置严格匹配
        res_pos = evaluate_entities(
            [data_true],
            [data_pred],
            mode="semi_structured",
            matching_method="position",
        )

        # 评测 2：文本严格匹配（exact）
        res_txt_exact = evaluate_entities(
            [data_true],
            [data_pred],
            mode="semi_structured",
            matching_method="text",
            text_match_mode="exact",
        )

        # 评测 3：文本重叠（overlap）
        res_txt_overlap = evaluate_entities(
            [data_true],
            [data_pred],
            mode="semi_structured",
            matching_method="text",
            text_match_mode="overlap",
        )

        # 异常样本记录（F1低于阈值时记录）
        if res_pos.get("f1_score", 0.9) < error_threshold:
            # 只保留完全不一致的对象
            true_entities = data_true.get("entities", [])
            pred_entities = data_pred.get("entities", [])
            
            # 构建完整实体标识（position + text）的映射
            true_full_map = {(e["start"], e["end"], e["text"]): e for e in true_entities}
            pred_full_map = {(e["start"], e["end"], e["text"]): e for e in pred_entities}
            
            # 找出完全不一致的entities（position或text任一不同）
            error_true_ents = []
            error_pred_ents = []
            matched_count = 0
            
            # 收集所有完整标识（union）
            all_full_keys = set(true_full_map.keys()) | set(pred_full_map.keys())
            
            for key in all_full_keys:
                true_ent = true_full_map.get(key)
                pred_ent = pred_full_map.get(key)
                
                # 完全相同（position和text都相同）
                if true_ent is not None and pred_ent is not None:
                    matched_count += 1
                    continue  # 跳过完全一致的
                
                # 不一致的情况
                if true_ent is not None:
                    error_true_ents.append(true_ent)
                if pred_ent is not None:
                    error_pred_ents.append(pred_ent)
            
            # 只在有错误实体时才写入
            if error_true_ents or error_pred_ents:
                with err_path.open("a", encoding="utf-8") as ef:
                    ef.write(
                        json.dumps(
                            {
                                "metrics": res_pos,
                                "ground_truth": {"entities": error_true_ents},
                                "predict": {"entities": error_pred_ents},
                                "report_title": report_title,
                                "report": report_text,  # 添加原始报告内容
                                "total_true": len(true_entities),
                                "total_pred": len(pred_entities),
                                "matched": matched_count,  # 完全匹配的数量
                                "error_count": len(error_true_ents) + len(error_pred_ents),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        # 累加（数值项求和，最后取平均）
        for k, v in res_pos.items():
            if isinstance(v, (int, float)):
                totals_pos[k] = totals_pos.get(k, 0.0) + float(v)
        for k, v in res_txt_exact.items():
            if isinstance(v, (int, float)):
                totals_txt_exact[k] = totals_txt_exact.get(k, 0.0) + float(v)
        for k, v in res_txt_overlap.items():
            if isinstance(v, (int, float)):
                totals_txt_overlap[k] = totals_txt_overlap.get(k, 0.0) + float(v)

        counted_reports += 1

    # 求平均（precision/recall/f1_score 三项）
    def _avg_inplace(d: Dict[str, float], n: int):
        if n <= 0:
            return
        for m in ("precision", "recall", "f1_score"):
            if m in d:
                d[m] = d[m] / n

    _avg_inplace(totals_pos, counted_reports)
    _avg_inplace(totals_txt_exact, counted_reports)
    _avg_inplace(totals_txt_overlap, counted_reports)

    print(
        f"[INFO] 计入平均的样本数 = {counted_reports}\n"
        f" - position     : {totals_pos}\n"
        f" - text (exact) : {totals_txt_exact}\n"
        f" - text (overlap): {totals_txt_overlap}"
    )
    return totals_pos, totals_txt_exact, totals_txt_overlap


# =========================
# 顶层：直接传参运行
# =========================
def run_evaluation(
    config_path: str,
    input_file: Optional[str] = None,
    align_mode: str = "gold",  # 'gold' | 'pred' | 'union'
    max_samples: int = 1000,
    max_report_tokens: int = 2000,
    seed: Optional[int] = 42,
    null_threshold: Optional[float] = None,
    error_dump: str = "data/test_semi_struct/error_positions.jsonl",
    report_types: Optional[List[str]] = None,  # 新增：指定报告类型
    error_threshold: float = 0.99,  # 新增：错误样本F1阈值
    exclude_keys: Optional[List[str]] = None,  # 新增：排除的key列表
) -> Dict[str, Dict[str, float]]:
    """
    直接传参的评估入口（无 argparse）。

    返回:
        {
          "position": {...},
          "text_exact": {...},
          "text_overlap": {...}
        }
    """
    from transformers import BertTokenizerFast as HFTokenizer

    # 读取共享配置
    cfgd = load_ebqa_cfg(config_path)

    # ==== 推理配置（从共享配置派生，可被函数参数覆盖） ====
    lens = lengths_from(cfgd)
    pred_blk = predict_block(cfgd)

    cfg = PredictConfig(
        report_struct_path=resolve_report_struct_path(cfgd),
        tokenizer_name=resolve_tokenizer_name(cfgd),
        model_dir=resolve_model_dir(cfgd),
        max_seq_len=lens["max_seq_len"],
        max_tokens_ctx=lens["max_tokens_ctx"],
        doc_stride=lens.get("doc_stride", 128),
        max_answer_len=lens["max_answer_len"],
        batch_size=int(pred_blk.get("batch_size", 1)),
        chunk_mode=chunk_mode_from(cfgd),
    )
    cfg.chunk_mode = chunk_mode_from(cfgd)

    cfg.batch_size = int(pred_blk["batch_size"])
    cfg.enable_no_answer = bool(pred_blk["enable_no_answer"])
    cfg.null_threshold = float(
        null_threshold if null_threshold is not None else pred_blk["null_threshold"]
    )
    cfg.null_agg = str(pred_blk["null_agg"])
    cfg.use_question_templates = bool(pred_blk["use_question_templates"])

    # 设随机种子
    set_seed(seed)

    # 加载模型与 collator
    model, collate, _ = load_ebqa(cfg)

    # tokenizer 用于 ground_truth 的 token 长度过滤
    tok = HFTokenizer.from_pretrained(cfg.tokenizer_name)
    print(f"[INFO] tokenizer: {cfg.tokenizer_name}")
    print(f"[INFO] struct: {cfg.report_struct_path}, model_dir: {cfg.model_dir}")

    # 构造 ground truth 子集（若未显式传入 input，就用配置里的）
    src_input = input_file or str(pred_blk["input_file"])
    gt_paths = ground_truth(
        src_input,
        tok,
        out_dir="data",
        max_samples=max_samples,
        max_report_tokens=max_report_tokens,
        seed=seed,
        struct_path=cfg.report_struct_path,
    )

    # 评测
    pos, txt_exact, txt_overlap = evaluate_model(
        model,
        collate,
        cfg,
        gt_paths["jsonl_path"],
        align_mode=align_mode,
        error_dump_path=error_dump,
        report_types_filter=report_types,  # 传递报告类型过滤
        error_threshold=error_threshold,  # 传递错误阈值
        exclude_keys=exclude_keys,  # 传递排除的key列表
    )

    # 可选：落盘一份 summary
    out_sum = Path("data/test_semi_struct/metrics_summary.json")
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    out_sum.write_text(
        json.dumps(
            {
                "position": pos,
                "text_exact": txt_exact,
                "text_overlap": txt_overlap,
                "config_path": config_path,
                "input_file": src_input,
                "align_mode": align_mode,
                "max_samples": max_samples,
                "max_report_tokens": max_report_tokens,
                "seed": seed,
                "null_threshold": cfg.null_threshold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] metrics saved -> {out_sum}")

    return {"position": pos, "text_exact": txt_exact, "text_overlap": txt_overlap}


# =========================
# 直接在 main 中运行
# =========================
if __name__ == "__main__":
    import os
    
    # 配置文件路径
    CONFIG_PATH = os.environ.get(
        "EBQA_CONFIG_PATH",
        "pre_struct/ebqa/ebqa_config.json",
    )
    
    # 读取配置文件
    cfgd = load_ebqa_cfg(CONFIG_PATH)
    pred_block = predict_block(cfgd)
    
    # 参数优先级：环境变量 > predict配置块 > 默认值
    INPUT_FILE = os.environ.get("EVAL_INPUT_FILE") or pred_block.get("input_file")
    ALIGN_MODE = pred_block.get("align_mode", "gold")
    MAX_SAMPLES = int(pred_block.get("max_eval_samples", 500))
    MAX_REPORT_TOKENS = int(pred_block.get("max_report_tokens", 2000))
    SEED = int(pred_block.get("seed", 42))
    NULL_THRESHOLD = pred_block.get("null_threshold")
    ERROR_DUMP = pred_block.get("error_dump", "data/test_semi_struct/error_positions.jsonl")
    ERROR_THRESHOLD = float(pred_block.get("error_threshold", 0.9))  # 错误样本F1阈值
    REPORT_TYPES = pred_block.get("eval_report_types")  # None=评估所有类型
    EXCLUDE_KEYS = pred_block.get("exclude_keys", ["无键名","其他"])  # 默认排除"无键名"
    
    print("=" * 80)
    print("评估参数")
    print("=" * 80)
    print(f"配置文件: {CONFIG_PATH}")
    print(f"输入文件: {INPUT_FILE}")
    print(f"对齐模式: {ALIGN_MODE}")
    print(f"最大样本数: {MAX_SAMPLES}")
    print(f"最大报告tokens: {MAX_REPORT_TOKENS}")
    print(f"随机种子: {SEED}")
    print(f"null阈值: {NULL_THRESHOLD if NULL_THRESHOLD is not None else '使用配置默认值'}")
    print(f"错误样本阈值: F1 < {ERROR_THRESHOLD}")
    print(f"错误样本输出: {ERROR_DUMP}")
    print(f"排除的key: {EXCLUDE_KEYS if EXCLUDE_KEYS else '无'}")
    print(f"报告类型过滤: {REPORT_TYPES if REPORT_TYPES else '所有类型'}")
    print("=" * 80)
    print()

    results = run_evaluation(
        config_path=CONFIG_PATH,
        input_file=INPUT_FILE,
        align_mode=ALIGN_MODE,
        max_samples=MAX_SAMPLES,
        max_report_tokens=MAX_REPORT_TOKENS,
        seed=SEED,
        null_threshold=NULL_THRESHOLD,
        error_dump=ERROR_DUMP,
        report_types=REPORT_TYPES,
        error_threshold=ERROR_THRESHOLD,
        exclude_keys=EXCLUDE_KEYS,
    )
    
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    print(json.dumps(results, ensure_ascii=False, indent=2))
