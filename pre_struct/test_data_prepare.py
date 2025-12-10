import sys

sys.path.append(".")
from pre_struct.data_aug.structs import get_report_structure_map
from pre_struct.ocr_split import split_ocr
from utils.baidu_ocr import ocr
# llm_client.py 中追加
import os
import json
import requests
from typing import Any, Dict, Optional, Union
from conf import API_KEY


def call_model_once(
    prompt: str,
    model: str = "qwen3-32b",
    base_url: str = "http://123.57.234.67:8000/v1",
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Any:
    # 优先使用入参，否则使用配置中的 API_KEY
    api_key = api_key or API_KEY
    url = f"{base_url.rstrip('/')}/model/single_report"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 简单重试：处理偶发 5xx/网络异常
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={"report": prompt},
                timeout=timeout,
            )
            # 对 5xx 进行重试
            if 500 <= resp.status_code < 600 and attempt < 2:
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as e:
            last_exc = e
            if attempt == 2:
                raise
            continue

    # 兼容两种返回结构: {"llm_ret": "..."} 或 {"data": {"llm_ret": "..."}}
    # 注意：上面循环保证了 data 已定义，若三次都失败会抛异常
    # 兼容两种返回结构: {"llm_ret": "..."} 或 {"data": {"llm_ret": "..."}}
    if isinstance(data, dict):
        if "llm_ret" in data:
            return data["llm_ret"]
        if isinstance(data.get("data"), dict) and "llm_ret" in data["data"]:
            return data["data"]["llm_ret"]
    # 兜底为字符串
    return data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)


def category_struct(ocr_text: str, report_structure_map):
    categories = list(report_structure_map.keys())
    categories_str = ",".join(categories)
    prompt_categort = f"""请根据以下报告内容，判断报告的类型：
    {ocr_text}
    报告类型可选项：{categories_str}
    请只返回报告的类型名称（必须是可选项之一），不要添加多余内容,如果报告类型不在可选项中,请返回"其他"  。
    输出格式:{{"category":"报告类型名称"}}
    """
    try:
        ret = call_model_once(prompt_categort)
        # ret 期望为 JSON 字符串
        category = json.loads(ret).get("category", "其他")
    except Exception:
        category = "其他"
    key_to_extract = report_structure_map.get(category, [])
    print(f"category:{category}")
    return category, key_to_extract


def semi_struct(ocr_text: str, key_to_extract):
    value_to_extract = {}
    for key in key_to_extract:
        print(key)
        prompt_semi_struct = f"""请你阅读以下报告内容，提取报告中与{key}相关的信息。
        输入文本:{ocr_text}
        请只返回提取到的信息，不要添加多余内容，也不要推理，直接截取与{key}相关的信息。
        输出格式:{{"{key}":"提取到的信息,从输入文本中原样不做改写的直接截取,没有提及输出NA"}}
        """
        try:
            ret = call_model_once(prompt_semi_struct)
            ret_json = json.loads(ret)
            if ret_json.get(key) not in (None, "NA", ""):
                value_to_extract.update(ret_json)
        except Exception:
            print(f"key: {key} 出错")
            value_to_extract[key] = "ERROR"
    return value_to_extract

def split_with_paragraph(ocr_resp, joiner: str = " ") -> str:
    """将 OCR 结果按段落拼接并返回单个字符串，段落之间以 "\n\n" 分隔。

    - 支持输入为单页(dict)或多页(list[dict])的百度 OCR 返回结构。
    - 若存在 `paragraphs_result`，按其 `words_result_idx` 顺序拼接段落；否则退化为将整页 `words_result` 视为一个段落。
    - 返回值类型为 `str`。
    """

    def _page_paragraph_texts(page: dict) -> list[str]:
        paragraphs = page.get("paragraphs_result", []) or []
        words_result = page.get("words_result", []) or []

        para_texts: list[str] = []
        if paragraphs:
            for para in paragraphs:
                idxs = para.get("words_result_idx", []) or []
                parts: list[str] = []
                for idx in idxs:
                    if isinstance(idx, int) and 0 <= idx < len(words_result):
                        w = words_result[idx]
                        text = w.get("words") if isinstance(w, dict) else None
                        if isinstance(text, str) and text:
                            parts.append(text)
                if parts:
                    para_texts.append(joiner.join(parts).strip())
        else:
            # 没有段落信息时，将整页 words_result 视为一个段落
            parts = []
            for w in words_result:
                text = w.get("words") if isinstance(w, dict) else None
                if isinstance(text, str) and text:
                    parts.append(text)
            if parts:
                para_texts.append(joiner.join(parts).strip())
        return para_texts

    if isinstance(ocr_resp, list):
        all_paras: list[str] = []
        for page in ocr_resp:
            if isinstance(page, dict):
                all_paras.extend(_page_paragraph_texts(page))
        return "\n\n".join([p for p in all_paras if p])
    elif isinstance(ocr_resp, dict):
        return "\n\n".join(_page_paragraph_texts(ocr_resp))
    else:
        raise TypeError(f"Unsupported ocr_resp type: {type(ocr_resp)}")


def _dump_json_atomic(path: str, data) -> None:
    """原子方式写入 JSON，逐条处理时也能安全落盘，避免中断丢失。

    写入流程：到同目录临时文件 -> flush+fsync -> os.replace 覆盖目标。
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp_path, path)
def main(report_structure_map, test_data_path):
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
        for item in test_data:
            # if (
            #     item["url"]
            #     != "https://files.ypb.plus/uploads/2024/09/2706397ab93f4cc08b2ab3eba45ea6d8.jpeg"
            # ):
            #     continue  
            if "url" not in item or not item["url"]:
                splited_ocr = item["report"]
            else:
                if item.get("baidu_ocr"):
                    ocr_resp = item["baidu_ocr"]
                    if isinstance(ocr_resp, str):
                        try:
                            ocr_resp = json.loads(ocr_resp)
                        except Exception:
                            pass
                else:
                    url = item["url"]
                    ocr_resp = ocr(url)
                item["baidu_ocr"] = json.dumps(ocr_resp, ensure_ascii=False)
                # 先统一得到文本版本
                splited_ocr = split_ocr(ocr_resp)

            # if item["report_title"]:
            #     report_title = item["report_title"]
            #     key_to_extract = report_structure_map.get(report_title, [])
            # else:
            report_title, key_to_extract = category_struct(
                splited_ocr, report_structure_map
            )
            if report_title == "其他":
                continue
            for key in item.keys():
                if key in key_to_extract:
                    key_to_extract.remove(key)    
            value_to_extract = semi_struct(splited_ocr, key_to_extract)
            item["report_title"] = report_title
            item["report"] = splited_ocr

            item.update(value_to_extract)
            # 每处理一条即保存一次，防止中断丢失
            _dump_json_atomic(test_data_path, test_data)
    # 结束后再次保存，确保最终状态
    _dump_json_atomic(test_data_path, test_data)
    return True


if __name__ == "__main__":
    report_structure_map = get_report_structure_map()
    test_data_path = "data/test_semi_struct/labeling_test_semi.json"
    main(report_structure_map, test_data_path)
