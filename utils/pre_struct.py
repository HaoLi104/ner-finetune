from openai import OpenAI
from pathlib import Path
import json
import re

client = OpenAI(
    base_url="https://qwen3.yoo.la/v1", api_key="haixin_csco1435tG8y98hTa717"
)


def pre_struct(prompt, text):
    # 步骤 1：调用模型提取小标题列表
    response = client.chat.completions.create(
        model="qwen3-32b",
        messages=[{"role": "user", "content": prompt + "\n" + text}],
    )
    titles = (
        response.choices[0]
        .message.content.strip()
        .replace("```json", "")
        .replace("```", "")
    )

    try:
        title_list = json.loads(titles).get("小标题")
    except Exception as e:
        print("解析标题失败:", e)
        return None

    if not title_list:
        print("未提取到小标题")
        return None

    # 步骤 2：根据标题列表，提取其在文本中的起止位置信息
    pattern = "|".join([re.escape(t) for t in title_list])
    blocks = re.split(f"({pattern})", text)

    position_list = []
    cursor = 0

    for i in range(1, len(blocks), 2):
        raw_title = blocks[i]
        title = raw_title.strip().replace(" ", "").replace("\n", "")
        content = blocks[i + 1].strip().lstrip(":：") if i + 1 < len(blocks) else ""

        title_search = re.search(re.escape(raw_title), text[cursor:])
        if title_search:
            start_pos = cursor + title_search.start()
            end_pos = cursor + title_search.end()

            position_list.append(
                {
                    "title": title,
                    "content": content,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                }
            )

            cursor = end_pos + len(content)

    # 步骤 3：切分原始文本为段落（包含前导内容）
    sections = split_text_with_leading(text, position_list)

    return sections


def split_text_with_leading(text, position_list):
    sections = []

    # 处理文本开头（在第一个标题之前的内容）
    if position_list and position_list[0]["start_pos"] > 0:
        first_start = position_list[0]["start_pos"]
        leading_text = text[0:first_start].strip()
        if leading_text:
            sections.append({"title": "", "text": leading_text})

    # 逐个标题段落切分
    for i, item in enumerate(position_list):
        start = item["start_pos"]
        end = (
            position_list[i + 1]["start_pos"]
            if i + 1 < len(position_list)
            else len(text)
        )
        section_text = text[start:end].strip()

        sections.append({"title": item["title"], "text": section_text})

    return sections


if __name__ == "__main__":
    text = Path("split_sample/sample.txt").read_text()
    prompt = (
        """请你阅读下面输入的病历报告OCR文本。
你的任务是：
1.判断这是一份什么类型的病历报告（如检验报告、检查报告、出院记录、手术记录等）。
2. 从报告中抽取出可以作为短标题的词语或短语,这些短标题将用于后续对报告进行切分。
3. 如果某个段落无法定义小标题，截取该段落前几个字作为小标题。
4. 按照以下格式输出：{'小标题': ['小标题1', '小标题2', ...]}
提取原则：
- 小标题应能代表报告中的结构化片段，如：日期、诊断信息、检查项目、治疗方案、检验指标等，不要包含患者个人信息(姓名、性别、年龄、身份证号等)、医院信息(医院名称、科室名称、住院号、床号等)。
- 小标题需贴合原文中实际出现的内容，不能随意修改或虚构。
-不需要对报告内容进行解析或结构化处理，仅提取可用于分段的标题。
输入病历报告OCR文本如下：
    """
        + text
    )
    res = pre_struct(prompt, text)
    for item in res:
        print("title:", item["title"])
        print("text:", item["text"])
        print("-" * 100)
