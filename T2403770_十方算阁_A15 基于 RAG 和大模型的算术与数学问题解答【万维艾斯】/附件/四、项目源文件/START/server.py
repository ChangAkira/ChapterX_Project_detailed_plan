import json
from flask import Flask, request, jsonify, stream_with_context, Response
import requests
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import csv
from dotenv import load_dotenv
from flask_cors import CORS
from collections import deque

load_dotenv()

app = Flask(__name__)
CORS(app)

# 从环境变量获取 DEMO 根目录
DEMO_ROOT = os.environ.get("DEMO_ROOT")
if not DEMO_ROOT:
    raise EnvironmentError("请设置系统环境变量 DEMO_ROOT 指向你的 项目 文件夹")

# 加载 Sentence-BERT 模型
model_path = os.path.join(DEMO_ROOT, 'models', 'all-MiniLM-L6-v2')
model = SentenceTransformer(model_path)

# 加载知识向量和 FAISS 索引
knowledge_vectors = []
vector_array = None
index = None

KNOWLEDGE_FILE = os.path.join(DEMO_ROOT, 'algebra_knowledge.CSV')

# 创建一个会话历史记录器，保存最近5轮对话
conversation_history = deque(maxlen=5)

def load_knowledge_base():
    global knowledge_vectors, vector_array, index

    knowledge_data = []
    try:
        with open(KNOWLEDGE_FILE, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                knowledge_data.append(row)
    except FileNotFoundError:
        print(f"错误：找不到知识库文件 {KNOWLEDGE_FILE}")
        return
    except Exception as e:
        print(f"加载知识库文件时发生错误: {e}")
        return

    knowledge_vectors.clear() # 清空之前的知识向量
    for item in knowledge_data:
        keywords = item['关键字']
        if keywords:
            vector = model.encode(keywords)
            knowledge_vectors.append({
                '分类': item['分类'],
                '关键字': keywords,
                '问题': item['问题'],
                '回答': item['回答'],
                '公式': item['公式'],
                '向量': vector.tolist()
            })

    if knowledge_vectors:
        vector_array = np.array([item['向量'] for item in knowledge_vectors]).astype('float32')
        dimension = vector_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vector_array)
    else:
        vector_array = None
        index = None

load_knowledge_base()
print("知识库加载完成！")

def search_knowledge_faiss(query_text, top_k=3):
    global knowledge_vectors, vector_array, index
    if index is None:
        print("错误：FAISS 索引未加载。")
        return []
    query_vector = model.encode(query_text).reshape(1, -1).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i in range(len(indices[0])):
        result_index = indices[0][i]
        if result_index != -1:
            results.append({
                '分类': knowledge_vectors[result_index]['分类'],
                '关键字': knowledge_vectors[result_index]['关键字'],
                '问题': knowledge_vectors[result_index]['问题'],
                '回答': knowledge_vectors[result_index]['回答'],
                '公式': knowledge_vectors[result_index]['公式'],
                '距离': distances[0][i]
            })
    return results

API_URL = os.environ.get("CHAT_API_URL", "http://localhost:11434/api/chat")
MODEL_NAME_STAGE1 = os.environ.get("MODEL_NAME_STAGE1", "deepseek-r1:latest")
MODEL_NAME_STAGE2 = os.environ.get("MODEL_NAME_STAGE2", "mightykatun/qwen2.5-math:1.5b")
PROMPT_STAGE1 = os.environ.get("PROMPT_STAGE1", "请从以下数学问题中提取关键术语和代数概念（只需输出词语，用空格分隔，使用中文）。如果是问候语或无数学内容，输出\"null\"。")

def generate(user_message=None):
    headers = {"Content-Type": "application/json"}
    global conversation_history

    if user_message:
        conversation_history.append({"role": "user", "content": user_message})
    else:
        user_message = request.json.get('message')
        if not user_message:
            yield f"data: {json.dumps({'error': 'message 参数缺失'})}\n\n"
            return
        conversation_history.append({"role": "user", "content": user_message})

    # --- Stage 1: 调用第一个大模型 生成检索 query ---
    data_stage1 = {
        "model": MODEL_NAME_STAGE1,
        "stream": False,
        "messages": [{"role": "user", "content": f"{PROMPT_STAGE1}\n\n用户输入：{user_message}"}]
    }
    try:
        response_stage1 = requests.post(API_URL, headers=headers, json=data_stage1)
        response_stage1.raise_for_status()
        response_json_stage1 = response_stage1.json()
        if 'message' in response_json_stage1 and 'content' in response_json_stage1['message']:
            search_query = response_json_stage1['message']['content'].strip()
            print(f"第一个大模型 (Qwen2.5-1.5B-Instruct) 的输出 (检索 Query): {search_query}")

            # --- Stage 2: 执行 RAG 流程 ---
            knowledge_results = search_knowledge_faiss(search_query, top_k=3)
            print("\n检索到的知识库内容:")
            relevant_knowledge = []
            threshold = 0.5  # 设置相关性阈值，可以根据实际效果调整
            if knowledge_results:
                for result in knowledge_results:
                    print(f"  关键字: {result['关键字']}, 距离: {result['距离']}")
                    if result['距离'] < threshold:
                        relevant_knowledge.append(result)
                        print("  (符合阈值，将被包含)")
                    else:
                        print("  (超出阈值，将被忽略)")
                    print("---")
            else:
                print("  没有检索到相关知识。")

            # --- Stage 3: 构建第二个大模型的 Prompt，包含对话历史 ---
            context_knowledge = ""
            if relevant_knowledge:
                for result in relevant_knowledge:
                    context_knowledge += f"分类: {result['分类']}\n问题: {result['问题']}\n回答: {result['回答']}\n公式: {result['公式']}\n---\n"

            # 准备对话历史文本
            history_text = ""
            for i, msg in enumerate(list(conversation_history)[:-1], 1):  # 不包括最新的用户消息
                if msg["role"] == "user":
                    history_text += f"用户: {msg['content']}\n"
                else:
                    history_text += f"助手: {msg['content']}\n"

            # 构建包含历史的提示
            prompt_stage2_input = f"""你是一位专业的代数问题解答助手，名字叫"第十章"。你基于开源大模型构建，擅长解答各类代数问题。

用中文交流，公式一律使用反斜杠和括号格式表示，全文不要使用markdown中的强调或标题语法。

首先，查看这些知识内容与我们的对话历史：

**---------知识库内容开始-----------**
{context_knowledge}
**---------知识库内容结束-----------**

**---------历史对话内容开始---------**
{history_text}
**---------历史对话内容结束---------**

我可能会问到我们历史对话内容中的问题。这些知识库内容供你参考，你需要更加详细清楚地回答我的问题。

查看完上面内容后，第一，参考知识库内容和历史对话内容，回答我的问题：**{user_message}**

第二，参考知识库内容向我简单推荐五个相关的知识点，每条之间换行；再推荐三本相关的书籍，每本之间换行。

第三，告诉我你参考了知识库中那些条目的名称。"""

            print("\n\n发送给第二个大模型 (Qwen2.5-7B-Instruct) 的完整 Prompt:")
            print(prompt_stage2_input)
            print("\n\n")

            # --- Stage 4: 调用第二个大模型 (mightykatun/qwen2.5-math:1.5b) 并流式输出 ---
            data_stage2 = {
                "model": MODEL_NAME_STAGE2,
                "stream": True,
                "messages": [{"role": "user", "content": prompt_stage2_input}]
            }

            # 用于收集完整的助手响应
            assistant_response_chunks = []

            with requests.post(API_URL, headers=headers, json=data_stage2, stream=True) as response_stage2:
                response_stage2.raise_for_status()
                for line in response_stage2.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        # 尝试解析JSON以获取内容
                        try:
                            line_data = json.loads(decoded_line.replace('data: ', ''))
                            if 'message' in line_data and 'content' in line_data['message']:
                                # 收集响应片段
                                assistant_response_chunks.append(line_data['message']['content'])
                        except:
                            pass  # 如果无法解析为JSON，忽略

                        # 向客户端发送流式数据
                        yield f"data: {decoded_line}\n\n"

            # 将完整的助手响应添加到会话历史
            try:
                full_response = ''.join(assistant_response_chunks)
                conversation_history.append({"role": "assistant", "content": full_response})
                print("助手回复已添加到会话历史")
            except Exception as e:
                print(f"添加助手回复到会话历史时出错: {e}")

        else:
            error_message = {"error": "解析第一个模型响应出错", "details": "响应缺少 'message' 或 'content' 字段", "response": response_json_stage1}
            yield f"data: {json.dumps(error_message)}\n\n"
    except requests.exceptions.RequestException as e:
        error_message = {"error": "API请求出错", "details": str(e)}
        yield f"data: {json.dumps(error_message)}\n\n"
    except json.JSONDecodeError as e:
        error_message = {"error": "解析第一个模型响应的JSON出错", "details": str(e), "response_text": response_stage1.text}
        yield f"data: {json.dumps(error_message)}\n\n"

@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_history

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "message 参数缺失"}), 400
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


url = "http://127.0.0.1:1224/api/ocr"
# 新增路由处理图片上传和 OCR
@app.route('/api/ocr', methods=['POST'])
def ocr():
    # 接收前端发送的Base64图片数据
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "image参数缺失"}), 400

    # 发送到本地OCR服务
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json={"base64": data['image']}, headers=headers)
        response.raise_for_status()
        res_dict = response.json()

        # 解析OCR响应并提取文本
        if res_dict.get('code') == 100 and res_dict.get('data') and len(res_dict['data']) > 0:
            ocr_text = res_dict['data'][0]['text']
            return Response(stream_with_context(generate(user_message=ocr_text)), mimetype='text/event-stream')
        else:
            return jsonify({"error": "OCR处理失败", "details": res_dict}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "OCR请求失败", "details": str(e)}), 500
    except (KeyError, IndexError) as e:
        return jsonify({"error": "OCR响应格式错误", "details": str(e)}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "OCR响应解析失败"}), 500

# 新增处理上传知识库文件的路由
@app.route('/api/upload_knowledge', methods=['POST'])
def upload_knowledge():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    print("接收到用户知识库")
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        try:
            file_content = file.read().decode('utf-8')
            reader = csv.reader(file_content.splitlines())
            header = next(reader)  # 读取上传文件的标题行
            expected_header = ['分类', '关键字', '问题', '回答', '公式']
            if header != expected_header:
                return jsonify({'error': f'文件格式不正确，标题行应为：{",".join(expected_header)}'}), 400

            with open(KNOWLEDGE_FILE, mode='a', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile)
                for row in reader:
                    writer.writerow(row)

            # 重新加载知识库
            load_knowledge_base()
            print("知识库已更新！请重启后端程序以应用更改。")
            return jsonify({'message': '知识库更新成功！请重启后端程序以应用更改。'}), 200
        except Exception as e:
            return jsonify({'error': f'处理文件时发生错误: {str(e)}'}), 500
    else:
        return jsonify({'error': '请上传 .csv 文件'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)