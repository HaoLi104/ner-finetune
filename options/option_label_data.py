import json
import sys

sys.path.append(".")

# 从标注数据的标签转换到业务标签
# 已标注的有 手术名称、手术部位、切口类型、淋巴结清扫要转化
surgery_options = {
    "日期": "yyyy-MM-dd",
    "术式": ["改良根治术", "经典根治术", "扩大根治术", "单纯切除术", "保乳切除术"],
    "切口类型": ["横梭形", "纵梭形", "斜梭形", "弧形", "其他"],
    "手术名称": [
        "保乳手术",
        "乳房切除术",
        "腋窝淋巴结清扫",
        "前哨淋巴活检",
        "乳房重建",
        "卵巢切除",
        "局部病灶姑息性手术",
    ],
    "手术类型": ["根治性手术", "姑息性手术", "其他辅助手术"],
    "手术部位": ["左乳", "右乳"],
    "淋巴结清扫": ["I组", "II组", "III组", "内乳淋巴结", "锁骨上淋巴结"],
}

# 映射关系
mapping_dict = {
    "手术名称": ["术式", "手术类型", "手术名称"],
    "手术部位": ["手术部位"],
    "切口类型": ["切口类型"],
    "淋巴结清扫": ["淋巴结清扫"],
}


def convert_ner2option(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        ret_data = []

        for item in data:
            annotations_result = item["annotations"][0]["result"]
            if not annotations_result:
                continue
            for annotation in annotations_result:
                labels = annotation["value"]["labels"][0]
                value_text = annotation["value"]["text"]
                
        return ret_data

def generate_option_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

        for item in data:
            annotations_result = item["annotations"][0]["result"]
            if not annotations_result:
                continue
            for annotation in annotations_result:
                labels = annotation["value"]["labels"][0]
                if labels in mapping_dict.keys():
                    value_text = annotation["value"]["text"]
                    # TODO 准备训练数据时删除 ‘。标注:xxx’
                    to_label_text = f"{value_text}。标注:{mapping_dict[labels]},由'{labels}'映射"
                    # 每一条作为txt文件的一行
                    with open("data/train_option_label.txt", "a", encoding="utf-8") as f:
                        f.write(to_label_text + "\n")

if __name__ == "__main__":
    generate_option_data("data/train_label.json")
