#!/usr/bin/env python3
"""Merge multiple Label Studio JSON files and split into train/val/test.

Usage:
  python merge_and_split.py \
    --inputs /path/a.json /path/b.json ... \
    --output-dir data/kv_ner_prepared \
    --train-ratio 0.8 --val-ratio 0.1 --seed 42

- Expects each input to be a Label Studio export (list of tasks with annotations/result).
- Keeps only tasks that contain at least one labeled entity (labels under value.labels).
- Shuffles before splitting.
- Outputs train.json, val.json, test.json in Label Studio format (list of tasks).
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_ls_file(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def has_labels(task: Dict[str, Any]) -> bool:
    anns = task.get("annotations") or []
    for ann in anns:
        for res in ann.get("result", []):
            labels = res.get("value", {}).get("labels")
            if labels:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Label Studio JSON files to merge")
    ap.add_argument("--output-dir", required=True, help="Where to write train/val/test JSON")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tasks: List[Dict[str, Any]] = []
    for p in args.inputs:
        tasks = load_ls_file(Path(p))
        all_tasks.extend(tasks)
    # 过滤无标注样本
    filtered = [t for t in all_tasks if has_labels(t)]
    print(f"Loaded {len(all_tasks)} tasks, kept {len(filtered)} with labels")

    random.seed(args.seed)
    random.shuffle(filtered)

    n = len(filtered)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_set = filtered[:n_train]
    val_set = filtered[n_train:n_train + n_val]
    test_set = filtered[n_train + n_val:]

    (out_dir / "train.json").write_text(json.dumps(train_set, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "val.json").write_text(json.dumps(val_set, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "test.json").write_text(json.dumps(test_set, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved train/val/test to {out_dir} with sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")


if __name__ == "__main__":
    main()
