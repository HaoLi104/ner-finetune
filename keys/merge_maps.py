import json
from pathlib import Path
from typing import Dict, List

STRUCT_PATH = Path("keys/keys_merged.json")  # 读取时允许不存在，将在 main 中兼容
ALIAS_PATH = Path("keys/ALIAS_MAP_0919.json")
OUT_PATH = Path("keys/keys_merged.json")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")


def merge_maps(struct_map: Dict[str, List[str]], alias_map: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    """
    合并规则：
    - 以 STRUCTURE_MAP 的标题与键顺序为主；
    - 若 ALIAS_MAP 中存在结构表没有的键，则在该标题下的末尾追加（按 ALIAS_MAP 的出现顺序）；
    - 输出格式：{ 标题: { 键: {"别名": [...], "Q": ""} } }
    - Q 置空字符串
    """
    out: Dict[str, Dict[str, Dict[str, object]]] = {}

    titles = set(struct_map.keys()) | set(alias_map.keys())

    for title in titles:
        keys_struct: List[str] = list(struct_map.get(title, []) or [])
        alias_for_title: Dict[str, List[str]] = alias_map.get(title, {}) or {}

        # 先用结构表中的键顺序
        ordered_keys: List[str] = list(keys_struct)
        # 再追加 alias_map 中存在但结构表缺失的键
        for k in alias_for_title.keys():
            if k not in ordered_keys:
                ordered_keys.append(k)

        # 构建输出项
        title_obj: Dict[str, Dict[str, object]] = {}
        for k in ordered_keys:
            aliases = alias_for_title.get(k, [])
            # 规范为列表
            if not isinstance(aliases, list):
                try:
                    aliases = list(aliases)
                except Exception:
                    aliases = []
            title_obj[k] = {
                "别名": aliases,
                "Q": "",
            }
        out[title] = title_obj

    return out


def main():
    # 始终以原始 STRUCTURE_MAP 作为来源
    struct_map = load_json(Path("keys/STRUCTURE_MAP_0919.json"))
    alias_map = load_json(ALIAS_PATH)

    merged = merge_maps(struct_map, alias_map)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"[OK] Wrote merged keys -> {OUT_PATH}")


if __name__ == "__main__":
    main()
