# encoding:utf-8

import requests
import base64
import sys
sys.path.append(".")
from conf import get_access_token

"""
医疗检验报告单识别
"""

def ocr(img_path):
    # request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/medical_report_detection"
    # request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/medical_summary"
    # request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
    # request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/table"
    # 二进制方式打开图片文件
    # f = open(img_path, "rb")
    # img = base64.b64encode(f.read())
    params = {
        "url": img_path,
        "multidirectional_recognize": "true",
        "detect_direction": "true",
        "paragraph": "true",
        "detect_direction": "true",
    }

    request_url = request_url + "?access_token=" + get_access_token()
    headers = {"content-type": "application/x-www-form-urlencoded"}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        print(response.json())
    return response.json()

def extract_all_paragraph_objects(words_result):
    """
    提取所有段落的对象，保持数据原结构。

    :param words_result: 包含所有文字信息和段落信息的列表
    :return: 包含所有段落对象的列表
    """
    paragraphs_result = words_result.get('paragraphs_result', [])
    all_paragraph_objects = []

    for paragraph in paragraphs_result:
        words_indices = paragraph['words_result_idx']
        paragraph_objects = [words_result['words_result'][idx] for idx in words_indices]
        all_paragraph_objects.append(paragraph_objects)

    return all_paragraph_objects

if __name__=="__main__":
    img_path = "https://files.ypb.plus/uploads/2024/09/b8c43054d43f48f4a353fbebd93a27ef"
    res = ocr(img_path)
    # print("---------------------------")

    # paragraph = extract_all_paragraph_objects(res)
    # print(paragraph)
