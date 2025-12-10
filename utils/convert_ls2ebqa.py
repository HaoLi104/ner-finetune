from pathlib import Path
import json
from typing import List,Dict, Any
import re

ROOT = Path(__file__).resolve().parents[1]
def paragraph_ocr(ocr_text: str, all_keys: List[str]) -> str:
    """
    根据给定关键词列表，在较长 OCR 文本中插入段落分隔符 '\n\n'。
    规则：
      1) 若文本长度 < 500，原样返回；
      2) 文本长度 >= 500，仅在匹配到的“关键字段名”处（尽量）作为段首插入 '\n\n'；
      3) 为“尽量少的段落”，对过密的断点进行稀疏化（默认相邻断点至少间隔 MIN_GAP 个字符）；
      4) 关键字段名做“字面值精确匹配”，但允许其后紧跟可选空白与中英/全半角冒号（:，：）。

    参数：
      - ocr_text: OCR 大段文本
      - all_keys: 关键词列表（字符串的“字面值”），会按原样精确匹配

    返回：
      - 插入了 '\n\n' 的文本
    """
    if not isinstance(ocr_text, str) or not ocr_text:
        return ocr_text
    if len(ocr_text) < 500:
        return ocr_text

    # ------ 可调参数：相邻断点最小间隔，越大 => 段落越长、段数越少 ------
    MIN_GAP = 500

    # 为空直接返回
    if not all_keys:
        return ocr_text

    # 去重与按长度降序（避免短 key 落在长 key 内部时重复命中）
    uniq_keys = sorted(
        set(k for k in all_keys if isinstance(k, str) and k.strip()),
        key=len,
        reverse=True,
    )

    # 构造正则：
    # 精确匹配 key 本体（转义），并允许其后有可选的空白 + 冒号（: 或 ：）
    # 使用捕获开头索引：match.start()
    key_alts = "|".join(re.escape(k) for k in uniq_keys)
    # (?:"|') 不参与，这里按字面值匹配；允许 key 后面接空白+冒号（可选）
    pattern = re.compile(rf"(?:{key_alts})(?:\s*[：:])?", flags=re.UNICODE)

    # 收集所有命中的起始位置
    hit_positions: List[int] = [m.start() for m in pattern.finditer(ocr_text)]
    if not hit_positions:
        return ocr_text

    # 对命中位置排序并去重
    hit_positions = sorted(set(hit_positions))

    # 稀疏化：保证相邻断点至少间隔 MIN_GAP
    sparse_breaks: List[int] = []
    last = -(10**9)
    for pos in hit_positions:
        if pos - last >= MIN_GAP:
            sparse_breaks.append(pos)
            last = pos

    # 若第一段不是从 0 开始且首断点距离开头很近，就不在最开头再人为加断点；
    # 这里仅在 sparse_breaks 指定的位置“前面”插入换行
    # 为了插入稳定，按从后往前插入，避免索引偏移
    parts: List[str] = []
    prev = 0
    for b in sparse_breaks:
        # 如果当前位置已经是段首（已有换行且为双换行），则跳过
        # 判断 b 前是否已有两个换行
        if b >= 2 and ocr_text[b - 2 : b] == "\n\n":
            continue
        parts.append(ocr_text[prev:b])
        parts.append("\n\n")
        prev = b
    parts.append(ocr_text[prev:])

    return "".join(parts)


def generate_all_keys(item_data):
    questions = item_data["questions"]
    category = item_data["category"]
    all_keys = []
    alias_path = ROOT / "pre_struct/ALIAS_MAPPING_18k.json"
    alias_data = json.loads(alias_path.read_text(encoding="utf-8"))
    category_alias = alias_data[category]
    for question in questions:
        key = question["value"]
        all_keys.append(key)
        if category_alias.get(key):
            all_keys.extend(category_alias[key])
    return all_keys


def convert(ls_data_path) -> List[Dict[str, Any]]:
    in_path = (ROOT / ls_data_path) if isinstance(ls_data_path, str) else ls_data_path
    data = json.loads(in_path.read_text(encoding="utf-8"))

    res = []
    for item in data:
        anno_k_v = {}
        item_data = item["data"]
        if item_data["category"] not in ["出院记录","门诊病历"]:
            continue
        keys = generate_all_keys(item_data)
        ocr_text = item_data["ocr_text"]
        annotations_result = item["annotations"][0]["result"]
        anno_k_v["report_title"] = item_data["category"]
        anno_k_v["report"] = paragraph_ocr(ocr_text, keys)
        for item_anno in annotations_result:
            anno_value = item_anno["value"]
            anno_k_v[anno_value["labels"][0]] = anno_value["text"]
        res.append(anno_k_v)

    out_path = ROOT / "data/ls2ebqa_converted.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入：{out_path.resolve()}，共 {len(res)} 条")
    return res


if __name__ == "__main__":
    ls_data_path = "data/tmp_raw_ls_data.json"
    convert(ls_data_path)
