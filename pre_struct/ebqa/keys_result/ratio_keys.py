# -*- coding: utf-8 -*-
from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Dict, List, Set, Any, Iterable, Optional, Tuple

# ===================== 可配置 =====================
# 目标：总共输出 5 轮（r0 + r1..r4），r1..r4 的 NKR 依次为 50%、40%、30%、20%
NKR_TARGETS = [0.50, 0.40, 0.30, 0.20]  # ΔK / K_cur

# 起始选择（三选一；按优先级生效）：
START_HINT_ABS: Optional[int] = None  # 例如 100（“先取 100 个”）
START_HINT_RATIO: Optional[float] = None  # 例如 0.30（“先取 30% 全局键”）
AUTO_START_MARGIN = 0.95  # 自动起点安全系数，避免四舍五入导致超限
CLAMP_TO_SAFE_START = True  # True=若 start 超出“可跑满5轮”的上限，则自动下调；False=不下调（可能提前吃光）

# 采样/输出
SEED_BASE = 2025
OUT_DIR = Path("out_scheme2")
MIN_REPORTS_PER_GROUP = 2
NEW_DATA_FRACTION = 1.0

# 数据/结构文件路径
PATH_STRUCT = Path("keys/keys_merged.json")
PATH_DATA = Path("data/clean_ocr_ppt_da_v4_3_report_drop_keys.json")
# ==================================================

RESERVED_FIELDS = {"report", "report_title", "report_titles", "meta"}


# ---------------- 工具 ----------------
def _uniq(xs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _rtype(rec: Dict[str, Any]) -> Optional[str]:
    return rec.get("report_title") or rec.get("报告类型") or rec.get("报告名称")


def _cand_by_type(struct_map: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    return {
        rtype: {k for k in _uniq(keys) if k not in RESERVED_FIELDS}
        for rtype, keys in struct_map.items()
    }


def _global_union(sets: Iterable[Set[str]]) -> Set[str]:
    s: Set[str] = set()
    for st in sets:
        s |= st
    return s


def _count_global_total_keys(struct_map: Dict[str, List[str]]) -> int:
    return len(_global_union(_cand_by_type(struct_map).values()))


def _size_selected_global(sel: Dict[str, Set[str]]) -> int:
    return len(_global_union(sel.values()))


# -------- 起点计算：保证“能跑满 NKR_TARGETS 的轮数”的最大起点 --------
def _max_safe_start_ratio(nkr_targets: List[float]) -> float:
    """返回在无限键的理想条件下，为了不在最后一轮前吃光，r0 起点比例的上限。
    公式：s_max = Π(1 - t_i)（i=1..m），因为每轮 K 会放大为 K/(1-t_i)。"""
    prod = 1.0
    for t in nkr_targets:
        prod *= max(0.0, 1.0 - float(t))
    return max(0.0, min(1.0, prod))


# ---------- 选键：按“绝对数量”起始（r0） ----------
def select_keys_start_count(
    struct_map: Dict[str, List[str]],
    start_count: int,
    *,
    seed: int = 42,
) -> Dict[str, Set[str]]:
    start_count = max(0, int(start_count))
    rnd = random.Random(seed)
    cand_by_type = _cand_by_type(struct_map)
    global_cand_sorted = sorted(_global_union(cand_by_type.values()))
    rnd.shuffle(global_cand_sorted)
    selected = set(global_cand_sorted[:start_count])
    return {rtype: (cand & selected) for rtype, cand in cand_by_type.items()}


# ---------- 选键：按目标 NKR（ΔK / K_cur = t）推进 ----------
def select_keys_by_target_nkr(
    struct_map: Dict[str, List[str]],
    t_nkr: float,  # 目标 NKR ∈ [0,1]
    prev_selected: Dict[str, Set[str]],
    *,
    seed: int = 42,
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    rnd = random.Random(seed)
    cand_by_type = _cand_by_type(struct_map)
    global_cand_sorted = sorted(_global_union(cand_by_type.values()))
    global_cand = set(global_cand_sorted)

    prev_global: Set[str] = _global_union(prev_selected.values()) & global_cand
    remaining = [k for k in global_cand_sorted if k not in prev_global]

    n_prev = len(prev_global)
    n_remain = len(remaining)

    if n_remain == 0 or t_nkr <= 0.0:
        need_delta = 0
    elif t_nkr >= 1.0 - 1e-12:
        need_delta = n_remain
    else:
        # 关键：ΔK = t/(1-t) * K_prev
        need_delta = math.ceil((t_nkr / (1.0 - t_nkr)) * n_prev)
        need_delta = max(1, min(need_delta, n_remain))

    rnd.shuffle(remaining)
    delta_pick = set(remaining[:need_delta])

    cur_global = prev_global | delta_pick
    sel_cur_by_type = {
        rtype: (cand & cur_global) for rtype, cand in cand_by_type.items()
    }

    n_delta = len(delta_pick)
    n_cur = len(cur_global)
    achieved_nkr = (n_delta / n_cur) if n_cur > 0 else 0.0

    info = {
        "n_keys_prev_global": n_prev,
        "n_keys_delta_global": n_delta,
        "n_keys_cur_global": n_cur,
        "remaining_keys": n_remain - n_delta,
        "requested_new_key_ratio": float(t_nkr),
        "achieved_new_key_ratio": round(achieved_nkr, 6),
        "exhausted": (n_remain - n_delta == 0),
    }
    return sel_cur_by_type, info


# ---------- 构建数据 + 统计 ----------
def build_round_data_union_prev(
    records: List[Dict[str, Any]],
    sel_cur: Dict[str, Set[str]],
    *,
    sel_prev: Optional[Dict[str, Set[str]]] = None,
    new_data_fraction: float = 1.0,
    seed: int = 2025,
    keep_fields: Iterable[str] = ("report", "report_title"),
    drop_empty: bool = True,
    min_reports_per_group: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rnd = random.Random(seed)
    keep_fields = set(keep_fields or ())

    sel_prev = sel_prev or {}
    K_prev_by_type = {t: set(v) for t, v in sel_prev.items()}
    K_cur_by_type = {t: set(v) for t, v in sel_cur.items()}
    K_new_by_type = {
        t: K_cur_by_type.get(t, set()) - K_prev_by_type.get(t, set())
        for t in K_cur_by_type.keys()
    }

    prev_ids, new_ids = set(), set()
    rtype_by_idx: Dict[int, str] = {}
    for idx, rec in enumerate(records):
        rtype = _rtype(rec)
        if not rtype:
            continue
        rtype_by_idx[idx] = rtype
        for k in K_prev_by_type.get(rtype, set()):
            if k in rec and str(rec[k]).strip():
                prev_ids.add(idx)
                break
        for k in K_new_by_type.get(rtype, set()):
            if k in rec and str(rec[k]).strip():
                new_ids.add(idx)
                break

    new_ids_list = list(new_ids)
    rnd.shuffle(new_ids_list)
    take_new_n = (
        (
            len(new_ids_list)
            if new_data_fraction >= 1.0
            else max(1, math.ceil(len(new_ids_list) * max(0.0, new_data_fraction)))
        )
        if new_ids_list
        else 0
    )
    chosen_ids = prev_ids | set(new_ids_list[:take_new_n])

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for idx in sorted(chosen_ids):
        rec = records[idx]
        rtype = rtype_by_idx.get(idx)
        if not rtype:
            continue
        allow_keys = K_cur_by_type.get(rtype, set())
        kept: Dict[str, Any] = {k: rec[k] for k in keep_fields if k in rec}
        present = False
        for k in allow_keys:
            if k in rec and rec[k] not in (None, ""):
                kept[k] = rec[k]
                present = True
        if present or not drop_empty:
            candidates.append((rtype, kept))

    from collections import Counter

    cnt = Counter([rtype for rtype, _ in candidates])
    keep_types = {t for t, c in cnt.items() if c >= min_reports_per_group}
    dropped_types = {t: c for t, c in cnt.items() if c < min_reports_per_group}
    dataset = [sample for rtype, sample in candidates if rtype in keep_types]

    prev_global = _global_union(K_prev_by_type.values())
    cur_global = _global_union(K_cur_by_type.values())
    new_global = cur_global - prev_global
    achieved_nkr = (len(new_global) / len(cur_global)) if cur_global else 0.0

    stats = {
        "num_prev_candidates": len(prev_ids),
        "num_new_candidates": len(new_ids),
        "num_new_selected": min(take_new_n, len(new_ids)),
        "num_total_selected_before_group_filter": len(candidates),
        "num_total_selected_after_group_filter": len(dataset),
        "min_reports_per_group": min_reports_per_group,
        "num_groups_kept": len(keep_types),
        "num_groups_dropped": len(dropped_types),
        "groups_dropped_detail": dropped_types,
        "keys_prev_total": sum(len(v) for v in K_prev_by_type.values()),
        "keys_new_total": sum(len(v) for v in K_new_by_type.values()),
        "n_keys_prev_global": len(prev_global),
        "n_keys_cur_global": len(cur_global),
        "n_keys_new_global": len(new_global),
        "achieved_new_key_ratio_global": round(achieved_nkr, 6),
    }
    return dataset, stats


# ---------- 统计键 映射到中文 ----------
def _to_cn_stats(d: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "num_prev_candidates": "旧样本候选数",
        "num_new_candidates": "新增样本候选数",
        "num_new_selected": "新增样本入选数",
        "num_total_selected_before_group_filter": "分组过滤前样本总数",
        "num_total_selected_after_group_filter": "分组过滤后样本总数",
        "min_reports_per_group": "每组最少样本数",
        "num_groups_kept": "保留组数",
        "num_groups_dropped": "丢弃组数",
        "groups_dropped_detail": "丢弃组明细",
        "keys_prev_total": "旧键总数",
        "keys_new_total": "新增键总数",
        "n_keys_prev_global": "全局旧键数",
        "n_keys_cur_global": "全局当前键数",
        "n_keys_new_global": "全局新增键数",
        "achieved_new_key_ratio_global": "全局实际新增占比",
        "n_keys_delta_global": "全局本轮新增键数",
        "requested_new_key_ratio": "本轮请求新增占比",
        "achieved_new_key_ratio": "实际新增占比",
        "remaining_keys": "剩余键数",
        "exhausted": "剩余键是否不足",
        "plan": "方案",
        "round": "轮次",
        "init_ratio": "起始覆盖率",
        "init_count": "起始键数",
        "init_count_safe_cap": "起始键数上限(可跑满5轮)",
    }
    return {mapping.get(k, k): v for k, v in d.items()}


# ---------- 主流程：5 轮划分（r0 + r1..r4） ----------
def run_scheme2_5rounds(struct_map, records, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 计算全局唯一键总数
    k_total = _count_global_total_keys(struct_map)
    if k_total <= 0:
        raise RuntimeError("全局唯一键总数为 0，无法划分。")

    # 计算“能跑满 5 轮”的起始上限（比例 & 绝对数）
    safe_ratio = _max_safe_start_ratio(NKR_TARGETS)
    safe_count_cap = max(1, int(math.floor(safe_ratio * k_total * AUTO_START_MARGIN)))

    # 确定 r0 起始数量（优先级：绝对数提示 > 比例提示 > 自动）
    if START_HINT_ABS is not None:
        start_count_req = int(START_HINT_ABS)
    elif START_HINT_RATIO is not None:
        start_count_req = int(math.floor(float(START_HINT_RATIO) * k_total))
    else:
        start_count_req = safe_count_cap  # 自动选择尽量接近上限的起点

    start_count = start_count_req
    if CLAMP_TO_SAFE_START and start_count > safe_count_cap:
        start_count = safe_count_cap  # 下调，保证能跑满 5 轮

    # r0：按起始数量选键
    sel_prev = select_keys_start_count(struct_map, start_count, seed=SEED_BASE)
    data0, st0 = build_round_data_union_prev(
        records,
        sel_prev,
        sel_prev=None,
        new_data_fraction=NEW_DATA_FRACTION,
        seed=SEED_BASE,
        keep_fields=("report", "report_title"),
        min_reports_per_group=MIN_REPORTS_PER_GROUP,
    )
    (out_dir / "r0.selected_keys.json").write_text(
        json.dumps(
            {k: sorted(v) for k, v in sel_prev.items()}, ensure_ascii=False, indent=2
        ),
        "utf-8",
    )
    (out_dir / "r0.dataset.json").write_text(
        json.dumps(data0, ensure_ascii=False, indent=2), "utf-8"
    )
    st0_en = dict(st0)
    st0_en.update(
        {
            "plan": "scheme2",
            "round": 0,
            "init_ratio": round(start_count / k_total, 6),
            "init_count": start_count,
            "init_count_safe_cap": safe_count_cap,
        }
    )
    (out_dir / "r0.stats.json").write_text(
        json.dumps(_to_cn_stats(st0_en), ensure_ascii=False, indent=2), "utf-8"
    )

    # r1..r4：按 NKR 推进；若吃光则停止输出后续轮
    for i, t in enumerate(NKR_TARGETS, start=1):
        sel_cur, info = select_keys_by_target_nkr(
            struct_map, t_nkr=t, prev_selected=sel_prev, seed=SEED_BASE + i
        )

        # 如果这一轮“新增=0”，说明已经无法继续；直接停止，不输出本轮
        if info["n_keys_delta_global"] == 0:
            break

        data, st = build_round_data_union_prev(
            records,
            sel_cur,
            sel_prev=sel_prev,
            new_data_fraction=NEW_DATA_FRACTION,
            seed=SEED_BASE + i,
            keep_fields=("report", "report_title"),
            min_reports_per_group=MIN_REPORTS_PER_GROUP,
        )

        (out_dir / f"r{i}.selected_keys.json").write_text(
            json.dumps(
                {k: sorted(v) for k, v in sel_cur.items()}, ensure_ascii=False, indent=2
            ),
            "utf-8",
        )
        (out_dir / f"r{i}.dataset.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
        )

        st_en = dict(st)
        st_en.update(info)
        st_en.update({"plan": "scheme2", "round": i})
        (out_dir / f"r{i}.stats.json").write_text(
            json.dumps(_to_cn_stats(st_en), ensure_ascii=False, indent=2), "utf-8"
        )

        sel_prev = sel_cur

        # 若本轮吃光，停止输出后续轮
        if info.get("exhausted"):
            break


# ======================= 入口 =======================
if __name__ == "__main__":
    struct_map = json.loads(PATH_STRUCT.read_text(encoding="utf-8"))
    records = json.loads(PATH_DATA.read_text(encoding="utf-8"))

    run_scheme2_5rounds(struct_map, records, out_dir=OUT_DIR)
