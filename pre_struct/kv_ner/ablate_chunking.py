#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Sweep chunk_size and chunk_overlap for inference-time evaluation.

Example:
  python pre_struct/kv_ner/ablate_chunking.py \
    --config pre_struct/kv_ner/kv_ner_config.json \
    --test_data data/kv_ner_eval/val_eval.jsonl \
    --chunk_sizes 384 448 500 \
    --overlaps 16 32 64 \
    --max_samples 300

Writes a JSON summary and prints a compact leaderboard.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from . import config_io
from .data_utils import build_bio_label_list
from .modeling import BertCrfTokenClassifier
from .evaluate import evaluate_dataset

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _default_model_dir(cfg: Dict[str, Any]) -> str:
    train_block = cfg.get("train", {})
    out_dir = train_block.get("output_dir")
    if isinstance(out_dir, str) and out_dir:
        return str(Path(out_dir) / "best")
    # fallback to config.model_dir if present
    md = cfg.get("model_dir")
    if isinstance(md, str) and md:
        return md
    return "runs/kv_ner/best"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk-size/overlap ablation for KV-NER eval")
    p.add_argument("--config", type=str, default="pre_struct/kv_ner/kv_ner_config.json")
    p.add_argument("--model_dir", type=str, default=None)
    p.add_argument("--test_data", type=str, default=None, help="JSONL; falls back to config.predict.input_path")
    p.add_argument("--output_dir", type=str, default="runs/kv_ner_ablate")
    p.add_argument("--chunk_sizes", type=int, nargs="+", default=[384, 448, 500])
    p.add_argument("--overlaps", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--align_mode", type=str, default="gold", choices=["gold", "pred", "union"])
    p.add_argument("--exclude_keys", type=str, nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_io.load_config(args.config)

    # Resolve paths
    model_dir = args.model_dir or _default_model_dir(cfg)
    if args.test_data:
        test_data_path = args.test_data
    else:
        pred_block = cfg.get("predict", {})
        test_data_path = pred_block.get("input_path")
        if not test_data_path:
            raise ValueError("--test_data not provided and predict.input_path missing in config")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build label maps
    label_map = config_io.label_map_from(cfg)
    labels = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Load model/tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertCrfTokenClassifier.from_pretrained(model_dir).to(device)
    tok_dir = Path(model_dir) / "tokenizer"
    if tok_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_io.tokenizer_name_from(cfg))

    # Load test data
    test_data = []
    with Path(test_data_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    logger.info("Samples: %d | Model: %s", len(test_data), model_dir)
    max_len = config_io.max_seq_length(cfg)

    grid: List[Tuple[int, int]] = []
    for cs in args.chunk_sizes:
        for ov in args.overlaps:
            grid.append((int(cs), int(ov)))

    results: List[Dict[str, Any]] = []
    for cs, ov in grid:
        # Guard to avoid overflowing special tokens
        eff_cs = min(int(cs), max_len)
        logger.info("Evaluating cs=%d ov=%d (eff_cs=%d)", cs, ov, eff_cs)
        stats = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            test_data=test_data if not args.max_samples else test_data[: int(args.max_samples)],
            id2label=id2label,
            device=device,
            max_seq_length=max_len,
            chunk_size=eff_cs,
            chunk_overlap=int(ov),
            merge_adjacent_gap=int(cfg.get("merge_adjacent_gap", 2)),
            error_dump_path=str(out_dir / f"errors_cs{eff_cs}_ov{ov}.jsonl"),
            error_threshold=0.99,
            align_mode=args.align_mode,
            exclude_keys=args.exclude_keys,
            report_title_filter=None,
            value_attach_window=int(cfg.get("value_attach_window", 50)),
            value_same_line_only=bool(cfg.get("value_same_line_only", True)),
            adjust_boundaries=bool(cfg.get("adjust_boundaries", True)),
            adjust_max_shift=int(cfg.get("adjust_max_shift", 1)),
            adjust_chars=str(cfg.get("adjust_chars", " \t\u3000:：,，.;；。()（）[]【】{}<>%％-—–/\\")),
            max_samples=args.max_samples,
        )
        row = {
            "chunk_size": eff_cs,
            "chunk_overlap": int(ov),
            "text_exact": stats.get("text_exact", {}),
            "text_overlap": stats.get("text_overlap", {}),
            "text_exact_in_k": stats.get("text_exact_in_k", {}),
            "num_samples": stats.get("num_samples", 0),
        }
        results.append(row)

    # Sort by text_exact F1 desc
    results.sort(key=lambda r: float(r.get("text_exact", {}).get("f1_score", 0.0)), reverse=True)

    # Save JSON summary
    summary = {
        "config": args.config,
        "model_dir": model_dir,
        "test_data": test_data_path,
        "max_seq_length": max_len,
        "grid": results,
    }
    out_path = out_dir / "chunk_ablation_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print compact leaderboard
    logger.info("\n=== Leaderboard (by Text Exact F1) ===")
    for i, r in enumerate(results[:10], 1):
        te = r.get("text_exact", {})
        logger.info(
            "#%d cs=%d ov=%d | exact F1=%.4f P=%.4f R=%.4f | overlap F1=%.4f",
            i,
            r["chunk_size"],
            r["chunk_overlap"],
            float(te.get("f1_score", 0.0)),
            float(te.get("precision", 0.0)),
            float(te.get("recall", 0.0)),
            float(r.get("text_overlap", {}).get("f1_score", 0.0)),
        )
    logger.info("Saved summary to %s", out_path)


if __name__ == "__main__":
    main()

