import pandas as pd
import requests
import json
import os  # 用于环境变量


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # 请参考 DeepSeek 官方文档获取最新 API 地址


DEEPSEEK_API_KEY = "#####"   #该程序仅短时间使用，且只在本地使用，因此为了方便直接将API Key编码进程序中。
if not DEEPSEEK_API_KEY:
    raise EnvironmentError("请设置名为 'DEEPSEEK_API_KEY' 的环境变量，包含你的 DeepSeek API 密钥。")

# 模型名称 
MODEL_NAME = "deepseek-chat" 

def extract_keywords_from_content(question, answer):
    """
    使用 DeepSeek V3 API 从问题和回答中提取关键字。
    """
    prompt = f"请从以下问题和回答中提取关键术语和概念（只需输出词语，用空格分隔，使用中文）：\n\n问题：{question}\n\n回答：{answer}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content'].strip()
        return result
    except requests.exceptions.RequestException as e:
        print(f"与 DeepSeek API 通信时发生错误 (提取关键字): {e}")
        return ""
    except json.JSONDecodeError as e:
        print(f"解析 DeepSeek API 响应时发生错误 (提取关键字): {e}")
        return ""
    except KeyError as e:
        print(f"DeepSeek API 响应格式错误 (提取关键字): {e}")
        print(response.json())
        return ""

def process_excel(excel_file):
    """
    处理 Excel 文件，从“问题”和“回答”列提取关键字并写入“关键字”列，
    每处理 30 条数据保存一次。
    """
    try:
        df = pd.read_excel(excel_file)
        if '问题' not in df.columns or '回答' not in df.columns:
            print("Excel 文件中缺少名为 '问题' 或 '回答' 的列。")
            return

        df['关键字'] = ''  # 初始化“关键字”列

        total_rows = len(df)
        save_interval = 30
        processed_count = 0

        for index, row in df.iterrows():
            question = str(row['问题'])
            answer = str(row['回答'])

            keywords = extract_keywords_from_content(question, answer)

            df.loc[index, '关键字'] = keywords

            processed_count += 1

            if processed_count % save_interval == 0:
                df.to_excel(excel_file, index=False)
                print(f"已处理完 {processed_count} / {total_rows} 条数据，并保存到 Excel 文件。")

        # 处理完所有数据后，确保最后一次保存
        if processed_count > 0 and processed_count % save_interval != 0:
            df.to_excel(excel_file, index=False)
            print(f"已处理完所有 {total_rows} 条数据，并保存到 Excel 文件。")
        elif total_rows == 0:
            print("Excel 文件为空，没有数据需要处理。")

    except FileNotFoundError:
        print(f"找不到 Excel 文件 '{excel_file}'。")
    except Exception as e:
        print(f"处理 Excel 文件时发生错误: {e}")

if __name__ == "__main__":
    excel_file_path = input("请输入 Excel 文件的路径：")
    process_excel(excel_file_path)