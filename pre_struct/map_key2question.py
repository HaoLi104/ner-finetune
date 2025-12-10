import json
import random

def convert_key_to_question(category, key):
    mapping_json = json.load(open("keys/keys_merged.json"))
    # ★ 统一使用keys_merged.json，去掉入院记录的特殊分支
    category_data = mapping_json.get(category, {})
    # 检查是否有自定义Q字段
    key_data = category_data.get(key, {})
    alias_str = ""
    
    if isinstance(key_data, dict):
        custom_question = key_data.get("Q", "")
        alias = key_data.get("别名", [])
        
        if alias:
            # 根据别名数量决定如何处理
            if len(alias) >= 2:
                # 随机选2个
                alias_str = f",{key}别名有" + "、".join(random.sample(alias, 2)) + "等"
            elif len(alias) == 1:
                # 只有1个别名
                alias_str = f",{key}别名有{alias[0]}"
            # len(alias) == 0 的情况alias_str保持为空
        
        if custom_question:
            return custom_question + alias_str
    
    return f"找到文本中的{key}" + alias_str


if __name__ == "__main__":
    print(convert_key_to_question("入院记录", "科室"))
