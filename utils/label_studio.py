from label_studio_sdk.client import LabelStudio
import json
from typing import Optional

# 基础配置
LS_URL = "http://127.0.0.1:8080"
API_KEY = "dbdfade3911e26d7664f03967622b4d600a41149"

ls = LabelStudio(base_url=LS_URL, api_key=API_KEY)


def get_project_id_by_title(title: str) -> Optional[int]:
    # 列表 → 过滤首个同名项目，返回 id
    projects = ls.projects.list()
    for p in projects:
        if p.get("title") == title:
            return p["id"]
    return None


def import_tasks(project_id: int, tasks_path: str):
    # tasks.json: [{"data": {...}}, ...]
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    # SDK 会按 label_config 校验字段是否匹配
    project = ls.projects.get(id=project_id)
    return project.import_tasks(tasks)


def export_json(project_id: int, out_path: str, include_all: bool = False):
    project = ls.projects.get(id=project_id)
    data = project.export(export_type="JSON", download_all_tasks=include_all)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===== 示例 =====
# pid = get_project_id_by_title('病历抽取-问答式')
# import_tasks(pid, 'tasks.json')
# export_json(pid, 'export.json', include_all=True)
