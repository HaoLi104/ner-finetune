#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from da_core.dataset import EnhancedQADataset as AliasQADataset, QACollator as AliasCollator

DEFAULT_EVAL_DATA = "data/alias_qa_dataset.json"
DEFAULT_CONFIG_PATH = str(Path(__file__).with_name("merged_config.json"))

try:
    from model_path_conf import DEFAULT_MODEL_PATH, DEFAULT_TOKENIZER_PATH  # type: ignore
except Exception:
    DEFAULT_MODEL_PATH = "/data/ocean/model/google-bert/bert-base-multilingual-cased/"
    DEFAULT_TOKENIZER_PATH = DEFAULT_MODEL_PATH


def load_alias_config(path: Optional[str]) -> Dict:
    cfg_path = Path(path or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Alias config not found: {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("merged_config.json must be a JSON object")
    return data


@torch.no_grad()
def evaluate_alias_model(
    model_dir: str,
    tokenizer_dir: str,
    data_path: str,
    *,
    batch_size: int = 16,
    max_seq_len: int = 512,
    max_tokens_ctx: int = 480,
    max_answer_len: int = 64,
    chunk_mode: str = "newline",
    question_template: str | None = None,
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds_kwargs = dict(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_tokens_ctx=max_tokens_ctx,
        max_answer_len=max_answer_len,
        autobuild=True,
        show_progress=True,
        chunk_mode=chunk_mode,
    )
    if question_template:
        ds_kwargs["question_template"] = str(question_template)
    dataset = AliasQADataset(**ds_kwargs)
    if not dataset.samples:
        raise RuntimeError(f"No samples found in {data_path}")

    collator = AliasCollator()
    em_scores, f1_scores = [], []

    for idx in range(0, len(dataset), batch_size):
        batch_samples = [dataset[i] for i in range(idx, min(len(dataset), idx + batch_size))]
        batch = collator(batch_samples)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        start_logits = outputs.start_logits.detach().cpu().numpy()
        end_logits = outputs.end_logits.detach().cpu().numpy()
        gold_start = batch["start_positions"].detach().cpu().numpy()
        gold_end = batch["end_positions"].detach().cpu().numpy()

        for ps, pe, gs, ge in zip(start_logits.argmax(axis=-1), end_logits.argmax(axis=-1), gold_start, gold_end):
            ps = int(ps)
            pe = int(pe)
            gs = int(gs)
            ge = int(ge)
            if gs == 0 and ge == 0:
                em = 1.0 if (ps == 0 and pe == 0) else 0.0
                f1 = em
            elif ps == 0 and pe == 0:
                em = 0.0
                f1 = 0.0
            else:
                pred_tokens = set(range(ps, pe + 1))
                gold_tokens = set(range(gs, ge + 1))
                inter = pred_tokens & gold_tokens
                if not gold_tokens or not inter:
                    em = 0.0
                    f1 = 0.0
                else:
                    precision = len(inter) / len(pred_tokens) if pred_tokens else 0.0
                    recall = len(inter) / len(gold_tokens)
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if precision and recall
                        else 0.0
                    )
                    em = 1.0 if pred_tokens == gold_tokens else 0.0
            em_scores.append(em)
            f1_scores.append(f1)

    return {
        "exact_match": float(np.mean(em_scores) if em_scores else 0.0),
        "f1": float(np.mean(f1_scores) if f1_scores else 0.0),
        "count": len(em_scores),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate alias QA model on multiple samples")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    cfgd = load_alias_config(args.config)

    model_dir = cfgd.get("model_dir") or os.path.join(cfgd.get("output_dir", "runs/alias_title"), "best")
    tokenizer_dir = (
        cfgd.get("tokenizer_name_or_path")
        or cfgd.get("model_name_or_path")
        or DEFAULT_TOKENIZER_PATH
    )
    data_path = args.data or cfgd.get("eval_data_path") or DEFAULT_EVAL_DATA

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Evaluation data not found: {data_path}")

    # 统一问法来源：使用训练段的 question_template
    qtpl = None
    try:
        qtpl = (cfgd.get("train", {}) or {}).get("question_template")
    except Exception:
        qtpl = None

    metrics = evaluate_alias_model(
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        data_path=data_path,
        batch_size=int(cfgd.get("eval_batch_size") or cfgd.get("batch_size") or 16),
        max_seq_len=int(cfgd.get("max_seq_len", 512)),
        max_tokens_ctx=int(cfgd.get("max_tokens_ctx", 480)),
        max_answer_len=int(cfgd.get("max_answer_len", 64)),
        chunk_mode=str(cfgd.get("chunk_mode", "newline")),
        question_template=qtpl,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
