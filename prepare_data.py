import os
from pathlib import Path
import pandas as pd


def import_txt2excel(folder_path, excel_path):
    folder_path = Path(folder_path).expanduser()
    excel_path = Path(excel_path)

    # 尝试读取已有 Excel，否则新建
    if excel_path.exists():
        df = pd.read_excel(excel_path)
    else:
        df = pd.DataFrame(columns=["cleaned_text"])

    new_contents = []

    # 遍历所有以 U 开头的子目录
    for subdir in folder_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith("U"):
            for file in subdir.iterdir():
                # 只处理后缀为 .txt，且不是 _all.txt 或类似聚合文件
                if (
                    file.is_file() and
                    file.suffix.lower() == ".txt" and
                    not file.name.startswith("_all") and
                    not file.name.endswith("_all.txt")
                ):
                    try:
                        content = file.read_text(encoding="utf-8").strip()
                        if content:
                            new_contents.append(content)
                    except Exception as e:
                        print(f"⚠️ 无法读取 {file}: {e}")

    if new_contents:
        new_df = pd.DataFrame({"cleaned_text": new_contents})
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(excel_path, index=False)
        print(f"✅ 成功追加 {len(new_contents)} 条文本到 {excel_path}")
    else:
        print("❗未发现任何符合条件的 .txt 内容")


if __name__ == '__main__':
    import_txt2excel('~/Documents/data/MedicalNER/数据集/乳腺癌', 'breast_cancer.xlsx')