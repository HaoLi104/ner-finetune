# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import re
import statistics as stats
import sys

sys.path.append(".")
try:
    import model_path_conf as _mpc  # type: ignore

    _TOKENIZER_NAME = getattr(_mpc, "DEFAULT_TOKENIZER_PATH", None)
    if not _TOKENIZER_NAME:
        raise ValueError("DEFAULT_TOKENIZER_PATH is missing")
except Exception as exc:
    raise ImportError(
        "DEFAULT_TOKENIZER_PATH must be defined in model_path_conf.py"
    ) from exc
from test_json import test_demo

# ====== 配置 ======
_MAX_TOKENS = 500  # 若总 token 超过该阈值，“插入段落符号 \n\n”

try:
    from transformers import BertTokenizerFast  # type: ignore

    _TOKENIZER = BertTokenizerFast.from_pretrained(
        _TOKENIZER_NAME, local_files_only=True, trust_remote_code=True
    )
except Exception:
    _TOKENIZER = None


# ====== 数据模型 ======
@dataclass
class BBox:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height


@dataclass
class Line:
    text: str
    box: BBox
    prob: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    lines: List[Line]
    size: Tuple[int, int] = (0, 0)


# ====== 小工具 ======
def _median(xs: List[float], default: float) -> float:
    return stats.median(xs) if xs else default


def _char_width_est(lines: List[Line]) -> float:
    """估算字符宽（备用），当前流程不做行内合并，仅保留工具函数。"""
    cands = []
    for l in lines:
        n = max(1, len(l.text))
        if n >= 10:
            cands.append(l.box.width / n)
    if cands:
        return _median(cands, 16.0)
    H = _median([l.box.height for l in lines], 30.0)
    return max(8.0, 0.6 * H)


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    if _TOKENIZER is not None:
        try:
            return len(_TOKENIZER.tokenize(text))
        except Exception:
            pass
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+|[^\s\w]", text, re.UNICODE))


# ======（保留但当前不使用）行内合并/行级排序 ======
_OPENERS = set("([{（【\"'“‘")
_CLOSERS = set(")]}）】\"'”’.,，。!！?？:：;；、")


def _need_space(prev: str, nxt: str) -> bool:
    if not prev or not nxt:
        return False
    if prev[-1] in _OPENERS:
        return False
    if nxt[0] in _CLOSERS:
        return False
    if prev.endswith((" ", "\t")) or nxt.startswith((" ", "\t")):
        return False
    return True


def _merge_same_row(page: Page) -> Page:
    """保留实现，但当前流程不调用。"""
    ls = sorted(page.lines, key=lambda x: (x.box.top, x.box.left))
    if not ls:
        return page
    H = _median([l.box.height for l in ls], 30.0)
    CW = _char_width_est(ls)

    def yc(l: Line) -> float:
        return l.box.top + 0.5 * l.box.height

    merged: List[Line] = []
    cur: Optional[Line] = None
    for ln in ls:
        if cur is None:
            cur = ln
            continue
        same_row = abs(yc(ln) - yc(cur)) <= 0.45 * max(
            H, 0.5 * (cur.box.height + ln.box.height)
        )
        if not same_row:
            merged.append(cur)
            cur = ln
            continue
        gap = ln.box.left - cur.box.right
        if gap >= max(4.0 * CW, 24.0):  # 防跨栏
            merged.append(cur)
            cur = ln
            continue
        if gap > 0:
            cur.text += (
                " " if _need_space(cur.text, ln.text) and gap > 0.35 * CW else ""
            ) + ln.text
        else:
            cur.text += ln.text
        cur.box = BBox(
            left=cur.box.left,
            top=min(cur.box.top, ln.box.top),
            width=max(cur.box.right, ln.box.right) - cur.box.left,
            height=max(cur.box.bottom, ln.box.bottom) - min(cur.box.top, ln.box.top),
        )
    if cur:
        merged.append(cur)
    page.lines = merged
    return page


def _order_by_rows(lines: List[Line]) -> List[Line]:
    """保留实现，但当前流程不调用。"""
    if not lines:
        return []
    ls = sorted(lines, key=lambda x: x.box.top)
    H_med = _median([l.box.height for l in ls], 30.0)
    row_thresh = 0.5 * H_med

    def yc(l: Line) -> float:
        return l.box.top + 0.5 * l.box.height

    groups: List[List[Line]] = []
    cur: List[Line] = []
    base_y: Optional[float] = None
    for ln in ls:
        y = yc(ln)
        if base_y is None:
            base_y = y
            cur = [ln]
            continue
        if abs(y - base_y) <= row_thresh:
            cur.append(ln)
        else:
            groups.append(sorted(cur, key=lambda x: x.box.left))
            base_y = y
            cur = [ln]
    if cur:
        groups.append(sorted(cur, key=lambda x: x.box.left))
    out: List[Line] = []
    for g in groups:
        out.extend(g)
    return out


# ====== 版面量化（阈值用到的左右边距）======
def _compute_block_boundaries(lines: List[Line]) -> Tuple[float, float]:
    xs_left = [ln.box.left for ln in lines if ln.text and ln.text.strip()]
    xs_right = [ln.box.right for ln in lines if ln.text and ln.text.strip()]
    if not xs_left or not xs_right:
        return (0.0, 0.0)
    return (min(xs_left), max(xs_right))


def _left_gap(ln: Line, min_left: float) -> float:
    return max(0.0, ln.box.left - min_left)


def _right_gap(ln: Line, max_right: float) -> float:
    return max(0.0, max_right - ln.box.right)


# ====== 断点判定（仅用于“插入段落符号”的候选） ======
_NUM_RE = re.compile(
    r"^\s*("
    r"[0-9]+[\.、)]?"  # 1. / 1、 / 1)
    r"|[一二三四五六七八九十百千]+[、.)]?"  # 中文数字
    r"|[ivxlcdmIVXLCDM]+[.)]?"  # 罗马
    r")"
)


def _starts_with_numbers(text: str) -> bool:
    return bool(text and _NUM_RE.match(text))


def _should_break(
    cur: Line,
    nxt: Line,
    min_left: float,
    max_right: float,
    boundary_threshold: float,
    indent_threshold: float,
) -> bool:
    """
    候选断点（插入 \n\n）：
      A) 当前行右边距很大（行末留白大），或
      B) 当前行以中文句号结尾 且（下一行不缩进 或 下一行不是编号）
    """
    if not (cur and nxt and cur.text is not None and nxt.text is not None):
        return False
    ends_early = _right_gap(cur, max_right) > boundary_threshold
    cur_ends_sentence = cur.text.rstrip().endswith("。")
    not_indented = _left_gap(nxt, min_left) <= indent_threshold
    next_not_num = not _starts_with_numbers(nxt.text)
    return bool(ends_early or (cur_ends_sentence and (not_indented or next_not_num)))


# ====== 兜底：仅中文句号 ======
_PUNCT_SPLIT = re.compile(r"[。]")  # 只用中文句号


def _is_sentence_end(line: Line) -> bool:
    return bool(
        line and isinstance(line.text, str) and line.text.rstrip().endswith("。")
    )


# ====== 渲染辅助（带断点重叠去重） ======
def _render(lines: List[Line], joiner: str = " ") -> str:
    """用 joiner 连接所有行文本，并折叠连续空白为单空格。"""
    toks = [ln.text for ln in lines if ln.text is not None]
    txt = joiner.join(toks)
    return re.sub(r"[ \t]+", " ", txt).strip()


def _strip_overlap(
    left_text: str, right_text: str, max_win: int = 30, min_win: int = 6
) -> str:
    """
    去掉右段与左段末尾的重叠前缀（仅当右段以左段末尾子串开头时删除）。
    例：left='...脑转移。', right='脑转移。随后继续...' -> 右段删除 '脑转移。'
    """
    lt = (left_text or "").rstrip()
    rt = (right_text or "").lstrip()
    if not lt or not rt:
        return rt
    win = min(max_win, len(lt), len(rt))
    for k in range(win, min_win - 1, -1):  # 优先更长的匹配
        suffix = lt[-k:]
        if rt.startswith(suffix):
            return rt[k:]
    return rt


def _render_with_break(
    lines: List[Line], break_idx: int, joiner: str = " ", para_break: str = "\n\n"
) -> str:
    """
    在 boundary=break_idx（介于 i 与 i+1）处插入 para_break。
    先分别渲染左右片段，再在拼接前做一次边界重叠去重，避免换段后重复。
    """
    if not lines:
        return ""
    n = len(lines)
    if break_idx < 0 or break_idx >= n - 1:
        return _render(lines, joiner)

    left_lines = lines[: break_idx + 1]
    right_lines = lines[break_idx + 1 :]

    left_text = _render(left_lines, joiner=joiner).rstrip()
    right_text = _render(right_lines, joiner=joiner).lstrip()

    right_text = _strip_overlap(left_text, right_text, max_win=30, min_win=6)

    return f"{left_text}{para_break}{right_text}".strip()


# ====== 主流程：不截断，只在超长时插入一个 \n\n ======
def split_ocr(
    ocr_obj: Any,
    boundary_threshold: Optional[float] = None,
    indent_threshold: Optional[float] = None,
    joiner: str = " ",
) -> str:
    """
    - 不做物理行合并；不按 top 排序；**严格按 OCR 返回顺序**拼接；
    - 若总 token ≤ _MAX_TOKENS：直接返回（不插入段落）；
    - 若 > _MAX_TOKENS：保留全部文本，在“自然断点”处插入 **一个** 段落符号 \\n\\n（不截断、不丢文本）。
      * 自然断点优先序：靠近阈值位置的候选断点(_should_break) -> 句号边界 -> 退化为阈值附近的任意边界。
    """

    # 解析输入（严格按输入列表顺序）
    def _page_from_dict(d: Dict[str, Any]) -> Page:
        lines: List[Line] = []
        for item in d.get("words_result", []):  # 不排序，保持原序
            loc = item.get("location", {})
            lines.append(
                Line(
                    text=item.get("words", ""),
                    box=BBox(
                        float(loc.get("left", 0)),
                        float(loc.get("top", 0)),
                        float(loc.get("width", 0)),
                        float(loc.get("height", 0)),
                    ),
                    prob=(
                        float(item.get("probability", 1.0))
                        if isinstance(item.get("probability", 1.0), (int, float))
                        else 1.0
                    ),
                )
            )
        return Page(lines=lines)

    # 支持单页/多页：严格按输入页面顺序与各页内部顺序拼接；不插入页间占位
    ordered_all: List[Line] = []
    if isinstance(ocr_obj, list):
        for p in ocr_obj:
            pg = _page_from_dict(p)
            ordered_all.extend(pg.lines)
    elif isinstance(ocr_obj, dict):
        pg = _page_from_dict(ocr_obj)
        ordered_all.extend(pg.lines)
    else:
        raise ValueError("Unsupported OCR JSON type")

    # 初始渲染
    text_full = _render(ordered_all, joiner=joiner)
    total_tokens = _count_tokens(text_full)
    if total_tokens <= _MAX_TOKENS:
        return text_full  # 不做任何段落插入

    # 计算阈值（基于版心宽度）
    non_empty = [ln for ln in ordered_all if ln.text]
    min_left, max_right = _compute_block_boundaries(non_empty)
    block_width = max(1.0, max_right - min_left)
    b_th = boundary_threshold if boundary_threshold is not None else 0.20 * block_width
    i_th = indent_threshold if indent_threshold is not None else 0.12 * block_width
    b_th = max(b_th, 12.0)
    i_th = max(i_th, 8.0)

    # —— 找到“阈值附近”的边界 —— #
    cum = 0
    boundary_near = 0
    for i, ln in enumerate(ordered_all):
        if i > 0:
            cum += _count_tokens(joiner)
        cum += _count_tokens(ln.text or "")
        if cum >= _MAX_TOKENS:
            boundary_near = min(i, len(ordered_all) - 2)  # 边界位于 i 与 i+1 之间
            break
    else:
        boundary_near = min(len(ordered_all) - 2, max(0, len(ordered_all) // 2))

    # —— 构造候选断点集合，优先选自然断点 —— #
    N = len(ordered_all)

    def _is_candidate(j: int) -> bool:
        if j < 0 or j >= N - 1:
            return False
        cur, nxt = ordered_all[j], ordered_all[j + 1]
        if not (cur and nxt and (cur.text or nxt.text)):
            return False
        return _should_break(cur, nxt, min_left, max_right, b_th, i_th)

    def _is_period_boundary(j: int) -> bool:
        if j < 0 or j >= N - 1:
            return False
        return _is_sentence_end(ordered_all[j])

    search_radius = max(10, N)
    chosen = None

    # 先正向从 boundary_near 开始找候选断点
    for offset in range(0, search_radius):
        j = boundary_near + offset
        if j >= N - 1:
            break
        if _is_candidate(j):
            chosen = j
            break

    # 再反向
    if chosen is None:
        for offset in range(1, search_radius):
            j = boundary_near - offset
            if j < 0:
                break
            if _is_candidate(j):
                chosen = j
                break

    # 无候选时，退化为“中文句号”边界（先正向后反向）
    if chosen is None:
        for offset in range(0, search_radius):
            j = boundary_near + offset
            if j >= N - 1:
                break
            if _is_period_boundary(j):
                chosen = j
                break
    if chosen is None:
        for offset in range(1, search_radius):
            j = boundary_near - offset
            if j < 0:
                break
            if _is_period_boundary(j):
                chosen = j
                break

    # 最后兜底：就用 boundary_near
    if chosen is None:
        chosen = boundary_near

    # —— 渲染：在 chosen 处插入 \n\n（不丢任何文本，断点重叠去重） —— #
    return _render_with_break(ordered_all, chosen, joiner=joiner, para_break="\n\n")


# ====== demo ======
if __name__ == "__main__":
    res = split_ocr(ocr_obj=test_demo)
    print(
        {
            "len_tokens": _count_tokens(res),
            "res": res,
        }
    )
