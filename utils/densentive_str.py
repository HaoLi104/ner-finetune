
import re

# 正则直接替换key：身份证号 电话号码
# 根据冒号替换value：姓名

import sys
sys.path.append("..")
# 身份证号码的正则表达式 18位
id_card_pattern = re.compile(r"\d{17}[\dXx]")
id_card_reserved_prefix = 6
id_card_reserved_postfix = 4

# 手机号 11位 (\+86)?[ \+-]*
phone_number_pattern = re.compile(r"1\d{10}")
phone_number_reserved_prefix = 6
phone_number_reserved_postfix = 1

# 中文姓名 ,如需匹配医生姓名，添加：|(医[ ]*师)|(医[ ]*生)
name_cn_pattern = re.compile(r"(姓[ ]*名)|(签[ ]*名)")
# 匹配位置右方 及 TODO 下方
name_cn_right_pattern = re.compile(r"[\u4e00-\u9fa5]{1,5}")
# name_cn_right_pattern = re.compile(r"^[^\s\:：]+")

# 工作单位正则（如：工作单位: XXX、单位：XXX）
workplace_pattern = re.compile(r"(工作单位|单位)[\s:：]*([^\n\r，,。；;]{2,30})")
# 病案号正则（如：病案号: XXX）
case_number_pattern = re.compile(r"(病案号)[\s:：]*([A-Za-z0-9\-]{3,30})")
# 患者ID/ID号正则（如：患者ID: XXX、ID号: XXX）
patient_id_pattern = re.compile(r"(患者ID|ID号)[\s:：]*([A-Za-z0-9\-]{3,30})")

def desensitize_workplace(text: str):
    """
    脱敏工作单位，只保留前2位，后面用*代替
    """
    def repl(m):
        prefix = m.group(1)
        value = m.group(2)
        if len(value) > 2:
            masked = value[:2] + "*" * (len(value) - 2)
        else:
            masked = "*" * len(value)
        return f"{prefix}: {masked}"
    return workplace_pattern.sub(repl, text)

def desensitize_case_number(text: str):
    """
    脱敏病案号，只保留前2位和后2位
    """
    def repl(m):
        prefix = m.group(1)
        value = m.group(2)
        if len(value) > 4:
            masked = value[:2] + "*" * (len(value) - 4) + value[-2:]
        else:
            masked = "*" * len(value)
        return f"{prefix}: {masked}"
    return case_number_pattern.sub(repl, text)

def desensitize_patient_id(text: str):
    """
    脱敏患者ID/ID号，只保留前2位和后2位
    """
    def repl(m):
        prefix = m.group(1)
        value = m.group(2)
        if len(value) > 4:
            masked = value[:2] + "*" * (len(value) - 4) + value[-2:]
        else:
            masked = "*" * len(value)
        return f"{prefix}: {masked}"
    return patient_id_pattern.sub(repl, text)

def desensitize_id_card(text: str):
    # 查找匹配的身份证号码
    matches = id_card_pattern.findall(text)

    # 替换匹配的身份证号码的中间四位为"*"
    for match in matches:
        # print(match)
        stars_num = len(match) - id_card_reserved_prefix - id_card_reserved_postfix
        masked = (
            match[:id_card_reserved_prefix]
            + "".join(["*" for i in range(stars_num)])
            + match[-id_card_reserved_postfix:]
        )
        text = text.replace(match, masked)
    return text


## 需要在id后面
def desensitize_phone_number(text: str):
    # masked_text = re.sub(phone_number_pattern, '****', text)
    matches = phone_number_pattern.findall(text)
    for match in matches:
        stars_num = (
            len(match) - phone_number_reserved_prefix - phone_number_reserved_postfix
        )
        masked = (
            match[:phone_number_reserved_prefix]
            + "".join(["*" for i in range(stars_num)])
            + match[-phone_number_reserved_postfix:]
        )
        text = text.replace(match, masked)
    return text


def desensitize_name_cn(text: str):
    # matches = name_cn_pattern.findall(text)
    matches = re.finditer(name_cn_pattern, text)
    for match in matches:
        print(match)
        seg = match.group()
        if len(seg) < 1:
            continue
        print(f"{seg}")
        posi = text.find(seg, match.start())

        text_remain = text[posi + len(seg) :]
        match_remain = name_cn_right_pattern.search(text_remain)
        if match_remain:
            print(f"    {match_remain}")
            str_to_replace = match_remain.group()
            posi_remain = text_remain.find(str_to_replace)
            if posi_remain > 12:
                # 距离太远，可能无关
                continue
            str_replaced = "*"
            if len(str_to_replace) > 0:
                str_replaced = str_to_replace[0] + "".join(
                    ["*" for i in range(len(str_to_replace) - 1)]
                )

            text = (
                text[0 : posi + len(seg)]
                + text_remain[0:posi_remain]
                + str_replaced
                + text_remain[posi_remain + len(str_to_replace) :]
            )
    return text


def desensitize_address(text: str):
    # 匹配“联系地址：”后面可能有空白和换行，下一行是地址
    pattern = re.compile(r"(联系地址|地址)[：:][ \t\r\f\v]*\n?([^\n\r]*)", re.MULTILINE)

    def mask_addr(match):
        # 无论下一行有没有内容都脱敏
        return f"{match.group(1)}：*"

    return pattern.sub(mask_addr, text)


# 脱敏
#   身份证号
#   手机
#   姓名/医生/医师
def desensitize(text):
    masked_text = text
    masked_text = desensitize_case_number(masked_text)
    masked_text = desensitize_workplace(masked_text)
    masked_text = desensitize_patient_id(masked_text)
    masked_text = desensitize_id_card(masked_text)
    masked_text = desensitize_phone_number(masked_text)
    masked_text = desensitize_name_cn(masked_text)
    masked_text = desensitize_address(masked_text)
    return masked_text