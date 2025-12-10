from pathlib import Path
import json
import pandas as pd
from split import split_text
import sys
import difflib  # 用于文本相似度
import re
from rapidfuzz import fuzz
sys.path.append(".")
from utils.densentive_str import desensitize

EXCEL_FILE      = r"breast_cancer.xlsx"
ID_COL          = "id"
JSON_MAP_FILE   = "category_id_map.json"
OUTPUT_BASE_DIR = "label_data"

def text_similarity(text1: str, text2: str) -> float:
    """
    返回 0.0–1.0 之间的相似度。
    1) 去除空白并小写化
    2) 使用 rapidfuzz.token_sort_ratio 计算，可处理顺序不同的词
    """
    # 简单预处理：去掉空白，小写
    t1 = "".join(text1.split()).lower()
    t2 = "".join(text2.split()).lower()
    # token_sort_ratio 返回 0–100，除 100 得到 0.0–1.0
    return fuzz.token_sort_ratio(t1, t2) / 100.0

def export_txt_by_key(key_name: str) -> None:
    mapping_path = Path(JSON_MAP_FILE)

    if not mapping_path.exists():
        raise FileNotFoundError(f"找不到映射文件 {JSON_MAP_FILE}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    ids = mapping.get(key_name, [])
    if not ids:
        print(f"⚠️  分类『{key_name}』在 JSON 中没有 id，已跳过")
        return

    df = pd.read_excel(EXCEL_FILE)
    if "full_content" not in df.columns:
        raise ValueError("Excel 中缺少 full_content 列")

    use_index = ID_COL not in df.columns
    if use_index:
        df["_row_id"] = df.index
        id_field = "_row_id"
    else:
        id_field = ID_COL

    sel = df[df[id_field].isin(ids)]
    out_dir = Path(OUTPUT_BASE_DIR) / key_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in out_dir.glob("*.txt"):
        file.unlink()
    skipped = 0
    written_texts = []  # 用于保存已写出的 desensitize_content 用于比对

    for _, row in sel.iterrows():
        doc_id = row[id_field]
        full_content = str(row["full_content"])
        desensitize_content = desensitize(full_content)
        split_content = split_text(desensitize_content)
        content = "\n".join(split_content)

        # 与已导出内容比对相似度
        skip_flag = False
        for prev_content in written_texts:
            sim = text_similarity(content, prev_content)
            if sim >= 0.9:
                skip_flag = True
                break
        if skip_flag:
            skipped += 1
            continue

        # 不相似才写出，并加到已写内容列表
        (out_dir / f"{doc_id}.txt").write_text(content, encoding="utf-8")
        written_texts.append(content)

    print(f"✅ 已在 {out_dir} 写出 {len(sel) - skipped} 个 txt，跳过相似度>90%的 {skipped} 个")

if __name__ == "__main__":
    # 放疗、化疗、病情诊断与评估-初诊、复发确诊、活检病理、手术、内分泌治疗、辅助治疗
    export_txt_by_key("放疗记录")
