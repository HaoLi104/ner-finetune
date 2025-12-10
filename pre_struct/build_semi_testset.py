# pre_struct/build_testset.py
# -*- coding: utf-8 -*-
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# =============== 与 dataset.py 对齐的辅助函数 ===============


def normalize_label(label: str) -> str:
    """与 dataset.py 一致：小写、去 '_' 和 '-'、strip。"""
    return label.lower().replace("_", "").replace("-", "").strip()


def _extract_spans_and_relations(ann: List[dict]):
    """
    从一条样本的 annotations[0].result 中提取：
      - spans_by_id: {span_id: {'label','start','end','text'}}
      - relations:   [{'from': title_span_id, 'to': content_span_id}]
    读取逻辑与 dataset.py 保持一致：只识别 REPORTTITLE / MODULETITLE / MODULECONTENT 三类。
    """
    spans_by_id = {}
    relations = []
    for obj in ann:
        t = obj.get("type")
        if t == "labels":
            span_id = obj["id"]
            raw = normalize_label(obj["value"]["labels"][0])
            if "reporttitle" in raw:
                label = "REPORTTITLE"
            elif "moduletitle" in raw:
                label = "MODULETITLE"
            elif "modulecontent" in raw:
                label = "MODULECONTENT"
            else:
                # 其他标签忽略
                continue
            spans_by_id[span_id] = {
                "label": label,
                "start": obj["value"]["start"],
                "end": obj["value"]["end"],
                "text": obj["value"]["text"],
            }
        elif t == "relation":
            relations.append({"from": obj["from_id"], "to": obj["to_id"]})
    return spans_by_id, relations


def _title_content_map_for_item(item: dict) -> Dict[str, List[str]]:
    """
    对单条样本，基于 relation 生成 {moduletitle: [modulecontent, ...]}。
    没有关联到任何 title 的 content 归入 key=""。
    """
    ann = item["annotations"][0]["result"]
    spans_by_id, relations = _extract_spans_and_relations(ann)

    # title_id -> [content_id...]
    title2content_ids: Dict[str, List[str]] = {}
    seen_content_ids = set()

    for rel in relations:
        t_id, c_id = rel["from"], rel["to"]
        t = spans_by_id.get(t_id)
        c = spans_by_id.get(c_id)
        if not t or not c:
            continue
        if not (t["label"] == "MODULETITLE" and c["label"] == "MODULECONTENT"):
            continue
        title2content_ids.setdefault(t_id, []).append(c_id)
        seen_content_ids.add(c_id)

    # 映射成 title_text -> [content_text ...]（按出现顺序、去重）
    title2contents_text: Dict[str, List[str]] = {}

    for t_id, c_ids in title2content_ids.items():
        t_obj = spans_by_id.get(t_id)
        if not t_obj:
            continue
        t_text = (t_obj.get("text") or "").strip()

        buf: List[Tuple[int, str]] = []
        for cid in c_ids:
            c_obj = spans_by_id.get(cid)
            if not c_obj:
                continue
            buf.append((c_obj["start"], (c_obj.get("text") or "").strip()))

        buf.sort(key=lambda x: x[0])
        dedup_contents: List[str] = []
        seen = set()
        for _, s in buf:
            if s and s not in seen:
                dedup_contents.append(s)
                seen.add(s)

        title2contents_text.setdefault(t_text, []).extend(dedup_contents)

    # 未被任何 title 关联的 content → key=""
    unpaired: List[Tuple[int, str]] = []
    for sid, sp in spans_by_id.items():
        if sp.get("label") == "MODULECONTENT" and sid not in seen_content_ids:
            unpaired.append((sp["start"], (sp.get("text") or "").strip()))

    unpaired.sort(key=lambda x: x[0])
    dedup_unpaired: List[str] = []
    seen_u = set()
    for _, s in unpaired:
        if s and s not in seen_u:
            dedup_unpaired.append(s)
            seen_u.add(s)

    if dedup_unpaired:
        title2contents_text.setdefault("", []).extend(dedup_unpaired)

    # 收集并附加 report_title（按出现顺序去重）
    rt_buf: List[Tuple[int, str]] = []
    for sid, sp in spans_by_id.items():
        if sp.get("label") == "REPORTTITLE":
            rt_buf.append((sp["start"], (sp.get("text") or "").strip()))
    rt_buf.sort(key=lambda x: x[0])
    report_title: List[str] = []
    seen_rt = set()
    for _, s in rt_buf:
        if s and s not in seen_rt:
            report_title.append(s)
            seen_rt.add(s)
    title2contents_text["report_title"] = report_title

    return title2contents_text


def build_test_kv(
    pre_struct_test_path: str = "data/pre_struct_test.json",
    out_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    从 pre_struct_test.json 生成基于 relation 的 {moduletitle: [modulecontent,...]} 测试集映射。
    - 每个样本生成一个字典，最终输出为 List[Dict[str, List[str]]]。
    - 未配对的 content 归入 key=""。
    - 默认写回同目录 pre_struct_test_kv.json（可通过 out_path 指定自定义路径）。
    - 输出中每个对象包含：
        * `report`：该样本的原始全文 data["text"]
        * `report_title`：按出现顺序去重的报告题名列表（来自 REPORTTITLE 标注）
    返回写入信息（路径、样本数）。
    """
    p = Path(pre_struct_test_path)
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    kv_list: List[Dict[str, List[str]]] = []

    for item in data:
        kv = _title_content_map_for_item(item)
        # 将原始全文文本写入本条对象的 report 字段
        report_text = ""
        try:
            report_text = (item.get("data") or {}).get("text", "")
        except Exception:
            report_text = ""
        if isinstance(kv, dict):
            kv["report"] = report_text
        else:
            kv = {"report": report_text}
        kv_list.append(kv)

    if out_path is None:
        out_path = str(p.parent / "pre_struct_test_kv.json")
    Path(out_path).write_text(
        json.dumps(kv_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "kv_path": out_path,
        "num_docs": len(kv_list),
    }


# =============== CLI ===============
if __name__ == "__main__":
    test_path = "data/test.json"
    info_kv = build_test_kv(
        pre_struct_test_path=test_path,
        out_path=None,  # 写到同目录 pre_struct_test_kv.json
    )
    print(json.dumps(info_kv, ensure_ascii=False, indent=2))
