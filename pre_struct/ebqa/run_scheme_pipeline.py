#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""General EBQA pipeline for running multiple round schemes automatically."""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import sys

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1] if len(_THIS_DIR.parents) > 1 else _THIS_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pre_struct.ebqa.da_core.dataset import EnhancedQADataset
from pre_struct.ebqa.da_core.utils import _save_jsonl, split_train_test_balanced_by_title
from pre_struct.ebqa.train_ebqa import TrainConfig, train_loop
from pre_struct.ebqa.config_io import (
    load_config as load_ebqa_config,
    train_block as ebqa_train_block,
)

# 可选导入集中路径配置（存在则用于覆盖默认模型/分词器路径）
try:
    import model_path_conf as _mpc  # type: ignore

    _DEFAULT_MODEL_PATH = getattr(_mpc, "DEFAULT_MODEL_PATH", None)
    _DEFAULT_TOKENIZER_PATH = getattr(
        _mpc, "DEFAULT_TOKENIZER_PATH", _DEFAULT_MODEL_PATH
    )
except Exception:
    _DEFAULT_MODEL_PATH = None
    _DEFAULT_TOKENIZER_PATH = None


def _is_local_dir(p: Optional[str]) -> bool:
    try:
        return bool(p) and os.path.isdir(str(p))
    except Exception:
        return False


def _resolve_tokenizer_name(cfgd: dict) -> str:
    """Prefer a valid local tokenizer directory if available.

    Priority: cfg.tokenizer_name_or_path (if local) > model_path_conf > cfg.model_name_or_path (if local) > fallback string
    """
    cand_cfg_tok = cfgd.get("tokenizer_name_or_path")
    if _is_local_dir(cand_cfg_tok):
        return str(cand_cfg_tok)
    if _is_local_dir(_DEFAULT_TOKENIZER_PATH):
        return str(_DEFAULT_TOKENIZER_PATH)
    if _is_local_dir(cfgd.get("model_name_or_path")):
        return str(cfgd["model_name_or_path"])
    # fallbacks (may point to HF repo id; acceptable for online envs)
    return str(cand_cfg_tok or _DEFAULT_TOKENIZER_PATH or cfgd.get("model_name_or_path", ""))


# 若集中配置提供了本地 tokenizer 目录，作为环境变量兜底，避免回退到旧默认
if _DEFAULT_TOKENIZER_PATH and os.path.isdir(str(_DEFAULT_TOKENIZER_PATH)):
    os.environ.setdefault("HF_TOKENIZER_NAME", str(_DEFAULT_TOKENIZER_PATH))


ROUND_FILE_RE = re.compile(r"^(?P<round>r\d+)\.dataset\.json$")


@dataclass
class RoundSpec:
    scheme: str
    round_id: str
    dataset_path: Path
    selected_keys_path: Path
    precomputed_path: Path
    output_dir: Path


def _normalise_round(rid: str) -> str:
    rid = rid.strip()
    if not rid:
        raise ValueError("Empty round id")
    if not rid.startswith("r"):
        rid = f"r{rid}"
    if not rid[1:].isdigit():
        raise ValueError(f"Invalid round id: {rid}")
    return rid


def _dataset_sort_key(path: Path) -> tuple:
    match = ROUND_FILE_RE.match(path.name)
    if match:
        rid = match.group("round")
        return (0, int(rid[1:]))
    return (1, path.name)


def _discover_rounds(
    *,
    scheme_name: str,
    dataset_dir: Path,
    base_output_dir: Path,
    precomputed_root: Optional[Path],
    include_rounds: Optional[Sequence[str]] = None,
) -> List[RoundSpec]:
    dataset_paths = sorted(dataset_dir.glob("*.dataset.json"), key=_dataset_sort_key)
    if not dataset_paths:
        raise FileNotFoundError(f"No *.dataset.json files found in {dataset_dir}")

    include: Optional[set] = None
    if include_rounds:
        include = {_normalise_round(r) for r in include_rounds}

    specs: List[RoundSpec] = []
    for dataset_path in dataset_paths:
        match = ROUND_FILE_RE.match(dataset_path.name)
        if not match:
            continue
        round_id = match.group("round")
        if include and round_id not in include:
            continue

        selected_keys = dataset_path.with_name(f"{round_id}.selected_keys.json")
        if not selected_keys.exists():
            raise FileNotFoundError(f"Missing selected keys for {round_id}: {selected_keys}")

        if precomputed_root:
            precomputed_path = precomputed_root / scheme_name / f"{round_id}.dataset.jsonl"
        else:
            precomputed_path = dataset_path.with_suffix(".jsonl")

        output_dir = base_output_dir / round_id

        specs.append(
            RoundSpec(
                scheme=scheme_name,
                round_id=round_id,
                dataset_path=dataset_path,
                selected_keys_path=selected_keys,
                precomputed_path=precomputed_path,
                output_dir=output_dir,
            )
        )

    if include and not specs:
        requested = ", ".join(sorted(include))
        raise ValueError(f"Requested rounds not found in {dataset_dir}: {requested}")

    return specs


def _build_precomputed(
    spec: RoundSpec,
    tokenizer_name: str,
    lengths: Dict[str, int],
    chunk_mode: str,
    force: bool,
    data_path_override: Optional[Path] = None,
) -> int:
    spec.precomputed_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.precomputed_path.exists() and not force:
        return -1

    data_path = str(data_path_override or spec.dataset_path)

    ds = EnhancedQADataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_seq_len=int(lengths["max_seq_len"]),
        max_tokens_ctx=int(lengths["max_tokens_ctx"]),
        max_answer_len=int(lengths["max_answer_len"]),
        use_question_templates=True,
        keep_debug_fields=False,
        negative_downsample=1.0,
        seed=42,
        autobuild=True,
        show_progress=True,
        chunk_mode=chunk_mode,
        report_struct_path=str(spec.selected_keys_path),
        only_title_keys=True,
    )
    _save_jsonl(ds.samples, str(spec.precomputed_path))
    return len(ds.samples)


def _make_train_config(cfgd: Dict, tb: Dict, spec: RoundSpec) -> TrainConfig:
    cfg_local = dict(cfgd)
    tb_local = dict(tb)

    tb_local["data_path"] = str(spec.precomputed_path)
    tb_local["precomputed"] = True

    cfg_local["report_struct_path"] = str(spec.selected_keys_path)
    cfg_local["output_dir"] = str(spec.output_dir)
    cfg_local["model_dir"] = str(Path(spec.output_dir) / "best")

    tokenizer_name = cfg_local.get("tokenizer_name_or_path") or cfg_local["model_name_or_path"]

    return TrainConfig(
        data_path=str(tb_local["data_path"]),
        precomputed=bool(tb_local["precomputed"]),
        report_struct_path=str(cfg_local["report_struct_path"]),
        model_name_or_path=str(cfg_local["model_name_or_path"]),
        tokenizer_name_or_path=str(tokenizer_name),
        max_seq_len=int(cfg_local["max_seq_len"]),
        max_tokens_ctx=int(cfg_local["max_tokens_ctx"]),
        max_answer_len=int(cfg_local["max_answer_len"]),
        output_dir=str(cfg_local["output_dir"]),
        num_train_epochs=int(tb_local["num_train_epochs"]),
        per_device_batch_size=int(tb_local["per_device_batch_size"]),
        grad_accum_steps=int(tb_local["grad_accum_steps"]),
        learning_rate=float(tb_local["learning_rate"]),
        weight_decay=float(tb_local["weight_decay"]),
        warmup_ratio=float(tb_local["warmup_ratio"]),
        max_grad_norm=float(tb_local["max_grad_norm"]),
        num_workers=int(tb_local["num_workers"]),
        pin_memory=bool(tb_local["pin_memory"]),
        eval_ratio=float(tb_local["eval_ratio"]),
        save_every_epochs=int(tb_local["save_every_epochs"]),
        early_stopping_patience=int(tb_local["early_stopping_patience"]),
        early_stopping_min_delta=float(tb_local["early_stopping_min_delta"]),
        plot_update_every=int(tb_local["plot_update_every"]),
        metrics_filename=str(tb_local["metrics_filename"]),
        seed=int(tb_local["seed"]),
        device=str(tb_local["device"]),
        allow_tf32=bool(tb_local["allow_tf32"]),
        chunk_mode=str(cfg_local["chunk_mode"]),
        label_smoothing=float(tb_local.get("label_smoothing", 0.0)),
        null_margin=float(tb_local.get("null_margin", 0.0)),
        null_margin_weight=float(tb_local.get("null_margin_weight", 0.0)),
        use_weighted_sampler=bool(tb_local.get("use_weighted_sampler", False)),
    )


def _export_train_test(dataset_path: Path, test_ratio: float, seed: int) -> Dict[str, Path]:
    """按目录落盘：将 rX.dataset.json 切分并分别写到 train/ 与 test/ 目录。

    输出：
      - <dataset_dir>/train/rX.dataset.json
      - <dataset_dir>/test/rX.dataset.json
    返回 dict: {"train": path, "test": path}
    """
    base_dir = dataset_path.parent
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_json = train_dir / dataset_path.name
    test_json = test_dir / dataset_path.name

    if train_json.exists() and test_json.exists():
        return {"train": train_json, "test": test_json}

    summary = split_train_test_balanced_by_title(
        data_path=str(dataset_path),
        out_train_json=str(train_json),
        out_test_json=str(test_json),
        test_ratio=float(test_ratio),
        seed=int(seed),
    )
    print(f"[SPLIT] {dataset_path.name}: {summary}")
    return {"train": train_json, "test": test_json}




def _resolve_paths(
    user_values: Sequence[Optional[str]],
    default_values: Sequence[Optional[str]],
    *,
    allow_none: bool = False,
    label: str,
) -> List[Optional[str]]:
    if not user_values:
        values = list(default_values)
    else:
        values = list(user_values)
        if len(values) == 1 and len(default_values) > 1:
            values = values * len(default_values)
    if len(values) != len(default_values):
        raise ValueError(f"{label} 个数与 --dataset-dir 不匹配")
    if not allow_none and any(v is None for v in values):
        raise ValueError(f"{label} 不能为空")
    return values


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare and train EBQA across multiple scheme rounds")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to ebqa_config.json template",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        help="Dataset directory containing r*.dataset.json files (repeatable)",
    )
    parser.add_argument(
        "--rounds",
        nargs="*",
        help="Optional subset of round ids (e.g. r0 r1 2)",
    )
    parser.add_argument(
        "--precomputed-root",
        action="append",
        default=[],
        help="Directory to save generated *.jsonl files per scheme (repeatable)",
    )
    parser.add_argument(
        "--output-root",
        action="append",
        default=[],
        help="Base output directory for model artifacts per scheme (repeatable)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override pretrained model path",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Override tokenizer path",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip dataset precomputation step",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training step",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild datasets even if jsonl already exists",
    )
    parser.add_argument(
        "--export-train-test",
        action="store_true",
        default=True,
        help="Split each r*.dataset.json into *_train.json and *_test.json before building",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Test ratio for dataset split when --export-train-test is set",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for dataset split when --export-train-test is set",
    )
    args = parser.parse_args(argv)

    cfgd = load_ebqa_config(args.config)
    tb = ebqa_train_block(cfgd)

    # 统一模型/分词器路径优先级：CLI > config.json > model_path_conf.py
    if args.model_name:
        cfgd["model_name_or_path"] = args.model_name
    elif _is_local_dir(_DEFAULT_MODEL_PATH):
        cfgd["model_name_or_path"] = _DEFAULT_MODEL_PATH

    if args.tokenizer_name:
        cfgd["tokenizer_name_or_path"] = args.tokenizer_name
    else:
        tok_cfg = cfgd.get("tokenizer_name_or_path")
        if _is_local_dir(tok_cfg):
            cfgd["tokenizer_name_or_path"] = tok_cfg
        elif _is_local_dir(_DEFAULT_TOKENIZER_PATH):
            cfgd["tokenizer_name_or_path"] = _DEFAULT_TOKENIZER_PATH
        elif _is_local_dir(cfgd.get("model_name_or_path")):
            cfgd["tokenizer_name_or_path"] = cfgd["model_name_or_path"]
        else:
            cfgd["tokenizer_name_or_path"] = tok_cfg or _DEFAULT_TOKENIZER_PATH or cfgd.get("model_name_or_path")

    dataset_dirs = args.dataset_dir or ["out_scheme1"]
    scheme_names = [Path(p).name for p in dataset_dirs]

    n_scheme = len(dataset_dirs)
    default_output = Path(cfgd["output_dir"])
    default_output_roots = [
        str(default_output.with_name(f"{default_output.name}_{name}")) for name in scheme_names
    ]
    default_precomputed_roots = [None] * n_scheme

    try:
        output_roots = _resolve_paths(
            args.output_root,
            default_output_roots,
            label="--output-root",
        )
        precomputed_roots = _resolve_paths(
            args.precomputed_root,
            default_precomputed_roots,
            allow_none=True,
            label="--precomputed-root",
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    tokenizer_name = _resolve_tokenizer_name(cfgd)
    lengths = {
        "max_seq_len": int(cfgd["max_seq_len"]),
        "max_tokens_ctx": int(cfgd["max_tokens_ctx"]),
        "max_answer_len": int(cfgd["max_answer_len"]),
    }

    chunk_mode = str(cfgd["chunk_mode"])

    for idx, dataset_dir_str in enumerate(dataset_dirs):
        dataset_dir = Path(dataset_dir_str)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        scheme = scheme_names[idx]
        output_root = Path(output_roots[idx])
        pre_root = precomputed_roots[idx]
        pre_root_path = Path(pre_root) if pre_root else None

        specs = _discover_rounds(
            scheme_name=scheme,
            dataset_dir=dataset_dir,
            base_output_dir=output_root,
            precomputed_root=pre_root_path,
            include_rounds=args.rounds,
        )

        for spec in specs:
            print(f"=== {spec.scheme} / {spec.round_id} ===")

            data_json_override: Optional[Path] = None
            if args.export_train_test:
                paths = _export_train_test(spec.dataset_path, args.test_ratio, args.split_seed)
                data_json_override = paths["train"]

            if not args.skip_build:
                built = _build_precomputed(
                    spec,
                    tokenizer_name=tokenizer_name,
                    lengths=lengths,
                    chunk_mode=chunk_mode,
                    force=args.force,
                    data_path_override=data_json_override,
                )
                if built >= 0:
                    print(
                        f"[BUILD] {spec.scheme} {spec.round_id}: generated {built} samples -> {spec.precomputed_path}"
                    )
                else:
                    print(
                        f"[BUILD] {spec.scheme} {spec.round_id}: using existing samples at {spec.precomputed_path}"
                    )
            else:
                if not spec.precomputed_path.exists():
                    raise FileNotFoundError(
                        f"Precomputed dataset missing for {spec.scheme} {spec.round_id}: {spec.precomputed_path}"
                    )
                print(
                    f"[BUILD] {spec.scheme} {spec.round_id}: skipped (existing {spec.precomputed_path})"
                )

            if args.skip_train:
                continue

            cfg = _make_train_config(cfgd, tb, spec)
            print(
                f"[TRAIN] {spec.scheme} {spec.round_id}: data={cfg.data_path} output_dir={cfg.output_dir}"
            )
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            train_loop(cfg)


if __name__ == "__main__":
    main()
