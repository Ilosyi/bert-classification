from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import re
from config import *

# 加载macbert分类模型
classifier_tokenizer = BertTokenizer.from_pretrained("macbert_classifier")
classifier_model = BertForSequenceClassification.from_pretrained("macbert_classifier")

def classify_question(question):
    inputs = classifier_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=CLASSIFIER_MAX_LENGTH)
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "简单问题" if prediction == 0 else "复杂问题"

def call_llama_local(question):
    payload = {
        "prompt": f"User: {question}\nAssistant:",
        "n_predict": LOCAL_MODEL_MAX_TOKENS,
        "temperature": 0.8,       # 控制随机性（默认值可能过高）
    "top_k": 40,              # 限制候选 token 数量
    "top_p": 0.95,             # Nucleus 采样
    "min_p": 0.05,
    "repeat_penalty": 1,    # 抑制重复
"stop": ["###", "User:"],
    }
    try:
        print(f"[DEBUG] 发送给本地模型的内容: {payload}")
        resp = requests.post(LOCAL_MODEL_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        print(f"[DEBUG] 本地模型返回内容: {data.get('content')}")
        return data.get("content", "[本地模型无返回内容]")
    except Exception as e:
        return f"[本地模型调用失败: {e}]"

def call_deepseek_api(question):
    """
    调用DeepSeek API处理复杂问题
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    system_prompt = """你是一个专业的助手。请用简洁、准确的方式回答用户的问题。
如果问题涉及技术实现，请提供具体的步骤和代码示例。
如果问题涉及分析或评价，请提供清晰的逻辑和观点。"""
    
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "stream": False,
        "temperature": DEEPSEEK_TEMPERATURE,
        "max_tokens": DEEPSEEK_MAX_TOKENS
    }
    
    try:
        print(f"[DEBUG] 发送给DeepSeek API的问题: {question}")
        response = requests.post(DEEPSEEK_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            print(f"[DEBUG] DeepSeek API返回内容: {content}")
            return content
        else:
            error_msg = f"DeepSeek API请求失败，错误码：{response.status_code}"
            print(f"[ERROR] {error_msg}")
            return f"[云端模型调用失败: {error_msg}]"
            
    except Exception as e:
        error_msg = f"DeepSeek API调用异常：{e}"
        print(f"[ERROR] {error_msg}")
        return f"[云端模型调用失败: {error_msg}]"

def extract_thought_and_answer(text):
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thought = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    else:
        thought = ""
        answer = text.strip()
    return thought, answer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/qa', methods=['POST'])
def qa():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    q_type = classify_question(question)
    if q_type == "简单问题":
        raw = call_llama_local(question)
        thought, answer = extract_thought_and_answer(raw)
    else:
        # 复杂问题调用DeepSeek API
        answer = call_deepseek_api(question)
        thought = "问题被判定为复杂问题，已由云端DeepSeek模型处理。"

    return jsonify({
        "question": question,
        "type": q_type,
        "thought": thought,
        "answer": answer
    })

if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
