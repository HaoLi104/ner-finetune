import pandas as pd
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.append(".")
from utils.densentive_str import desensitize


def calculate_similarity(text1, text2):
    """计算两个文本的余弦相似度"""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]


def process_breast_cancer_data(
    excel_file, output_folder, keywords, similarity_threshold=0.7, max_new_files=10
):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取当前日期和时间用于文件名
    current_date = datetime.now().strftime("%Y%m%d%H%M")

    # 读取现有文件内容
    existing_files = [f for f in os.listdir(output_folder) if f.endswith(".txt")]
    existing_contents = []
    for file in existing_files:
        with open(os.path.join(output_folder, file), "r", encoding="utf-8") as f:
            existing_contents.append(f.read())

    # 初始化新文件计数器
    new_files_written = 0

    # 读取Excel文件
    try:
        df = pd.read_excel(excel_file)

        # 检查full_content列是否存在
        if "full_content" not in df.columns:
            print("Error: 'full_content' column not found in the Excel file")
            return

        # 遍历full_content列
        for index, content in df["full_content"].items():
            # 检查是否已达到最大新文件数量
            if new_files_written >= max_new_files:
                print(f"Reached maximum new files limit ({max_new_files})")
                break

            # 检查内容是否为字符串且不为空
            if isinstance(content, str) and content.strip():
                # 检查是否同时包含所有关键词
                content_lower = content.lower()
                if all(keyword.lower() in content_lower for keyword in keywords):
                    # 检查相似度
                    is_unique = True
                    for existing_content in existing_contents:
                        similarity = calculate_similarity(content, existing_content)
                        if similarity > similarity_threshold:
                            is_unique = False
                            break

                    # 如果内容是独特的，写入新文件
                    if is_unique:
                        output_file = os.path.join(
                            output_folder, f"content_{index}_{current_date}.txt"
                        )
                        if not os.path.exists(output_file):
                            content_to_write = desensitize(content).replace("\n", "\\n")
                            with open(output_file, "w", encoding="utf-8") as f:
                                f.write(content_to_write)
                            existing_contents.append(content)
                            print(f"Written content to {output_file}")
                            new_files_written += 1

    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")

    print(f"Total new files written: {new_files_written}")


# 使用示例
if __name__ == "__main__":
    excel_file = "breast_cancer.xlsx"
    output_folder = "paragraph_data"
    keywords = ["出院记录"] 
    process_breast_cancer_data(
        excel_file, output_folder, keywords, similarity_threshold=0.7, max_new_files=10
    )
