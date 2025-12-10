# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import html
from pathlib import Path
from typing import Dict, Any, List, Iterable, Optional

# ====== 可按需修改的常量 ====== 转换alias为label studio所需格式
INPUT_PATH = "data/merged.converted.json"   # 你的原始记录文件（列表，每条含 report_title / report / alias）
OUT_DIR = "data/alias_ls/"                  # 输出目录
MAX_FILE_SIZE_MB = 30.0                      # 每个文件的最大大小（MB）
MAX_TOTAL: Optional[int] = None             # 本次最多导出多少条；None 表示不限制
ENCODING = "utf-8"


# ====== 小工具 ======
def _is_mapping(x) -> bool:
    return isinstance(x, dict)

def _as_str(x) -> str:
    if x is None:
        return ""
    return str(x)

def alias_to_table_html(alias: Dict[str, Any], rec: Dict[str, Any]) -> str:
    """把 alias 映射渲染成安全的 HTML 表格（供 <HyperText> 展示）。

    为了便于人工核对，这里同时展示：
    - 别名候选（来自 alias 映射的值）
    - 报告值（来自原始记录 rec[k]，即报告中抽取/识别的值）
    """
    rows: List[str] = []
    for k, v in alias.items():
        alias_str = _as_str(v)
        # 跳过空别名
        if alias_str == "":
            continue
        k_esc = html.escape(_as_str(k))
        alias_esc = html.escape(alias_str)
        # 对应的“报告值”，尽量直接从原记录取值
        report_val = _as_str(rec.get(k, ""))
        report_val_esc = html.escape(report_val)
        rows.append(f"<tr><th>{k_esc}</th><td>{alias_esc}</td><td>{report_val_esc}</td></tr>")
    # 注意：不要在这里放 <script> 或未转义的 &/"<' 等，以免触发解析错误
    table = (
        "<table class='kv-table'>"
        "<thead><tr><th>字段</th><th>别名候选</th><th>报告值</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )
    return table

def record_to_task(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    把一条原始记录转换成 Label Studio 任务：
    data = { report_title, report, alias_table_html, alias_pretty, alias_raw }
    """
    alias = rec.get("alias") or {}
    if not _is_mapping(alias):
        alias = {}
    
    # 过滤掉 value 为空字符串的项
    alias = {k: v for k, v in alias.items() if _as_str(v) != ""}

    report_title = _as_str(rec.get("report_title"))
    report = _as_str(rec.get("report"))

    data = {
        "report_title": report_title,
        "report": report,
        # 供 HyperText 直出预览
        "alias_table_html": alias_to_table_html(alias, rec),
        # 供 TextArea 直接编辑（人只改 value，不改 key）
        "alias_pretty": json.dumps(alias, ensure_ascii=False, indent=2),
        # 保留原始结构（训练/导出后处理可能用得上）
        "alias_raw": alias,
    }
    return {"data": data}

def chunk(lst: List[Any], n: int) -> Iterable[List[Any]]:
    n = max(1, int(n))
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def chunk_by_size(
    tasks: List[Dict[str, Any]], 
    max_size_bytes: int
) -> Iterable[List[Dict[str, Any]]]:
    """
    按照文件大小动态分割任务列表。
    
    Args:
        tasks: 任务列表
        max_size_bytes: 每个块的最大字节数
    
    Yields:
        任务块列表
    """
    if not tasks:
        return
    
    current_chunk = []
    current_size = 0
    # JSON数组的基础开销（方括号、换行等）
    json_overhead = 10
    
    for task in tasks:
        # 计算当前任务的大小（序列化为JSON）
        task_json = json.dumps(task, ensure_ascii=False, indent=2)
        task_size = len(task_json.encode('utf-8'))
        
        # 如果添加这个任务会超过限制，且当前块非空，则输出当前块
        estimated_size = current_size + task_size + json_overhead
        if current_chunk and estimated_size > max_size_bytes:
            yield current_chunk
            current_chunk = [task]
            current_size = task_size
        else:
            current_chunk.append(task)
            current_size += task_size
    
    # 输出最后一个块
    if current_chunk:
        yield current_chunk


# ====== 主流程 ======
def prepare_tasks_v2(
    input_path: str = INPUT_PATH,
    out_dir: str = OUT_DIR,
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
    max_total: Optional[int] = MAX_TOTAL,
) -> List[Path]:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src.resolve()}")

    # 读取原始记录（应为数组）
    records = json.loads(src.read_text(encoding=ENCODING))
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of record objects")

    if isinstance(max_total, int) and max_total > 0:
        records = records[:max_total]

    # 转换
    tasks: List[Dict[str, Any]] = []
    bad = 0
    for rec in records:
        try:
            tasks.append(record_to_task(rec))
        except Exception as e:
            bad += 1
            # 跳过坏样本，继续
            continue

    # 按文件大小动态分割任务
    max_size_bytes = int(max_file_size_mb * 1024 * 1024 * 0.95)  # 留5%余量
    print(f"[配置] 目标文件大小上限: {max_file_size_mb:.2f} MB (实际限制: {max_size_bytes / (1024 * 1024):.2f} MB)")

    # 写出（数组形式，LS 官方示例就是 [ {\"data\": {...}} ]）：
    # 你可在 Data → Import → Upload Files 直接导入。
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    total = len(tasks)
    
    # 使用动态大小分割
    chunks = list(chunk_by_size(tasks, max_size_bytes))
    nfiles = len(chunks)
    
    for idx, part in enumerate(chunks, start=1):
        out_path = out_root / f"tasks_{idx:04d}.json"
        content = json.dumps(part, ensure_ascii=False, indent=2)
        out_path.write_text(content, encoding=ENCODING)
        
        # 验证文件大小
        actual_size = out_path.stat().st_size
        size_mb = actual_size / (1024 * 1024)
        status = "✓" if size_mb < max_file_size_mb else "✗"
        print(f"  {status} {out_path.name}: {len(part)} 任务, {size_mb:.3f} MB")
        
        written.append(out_path)

    print(f"[OK] total={total}, bad={bad}, files={nfiles}, out_dir={out_root.resolve()}")
    if written:
        print(f"示例：导入 {written[0].name} 到 Label Studio（Data → Import → Upload Files）")
    return written


if __name__ == "__main__":
    prepare_tasks_v2()
