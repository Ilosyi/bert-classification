from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import re

# 加载macbert分类模型
classifier_tokenizer = BertTokenizer.from_pretrained("macbert_classifier")
classifier_model = BertForSequenceClassification.from_pretrained("macbert_classifier")

def classify_question(question):
    inputs = classifier_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "简单问题" if prediction == 0 else "复杂问题"

def call_llama_local(question):
    url = "http://127.0.0.1:8080/completion"
    payload = {
        "prompt": question,
        "n_predict": 128
    }
    try:
        print(f"[DEBUG] 发送给llama.cpp的内容: {payload}")
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        print(f"[DEBUG] llama.cpp返回内容: {data.get('content')}")
        return data.get("content", "[本地模型无返回内容]")
    except Exception as e:
        return f"[本地模型调用失败: {e}]"

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
        thought = "问题被判定为复杂问题，需由云端模型处理（当前未部署）。"
        answer = "[info] 复杂问题将由云端模型处理（当前未部署）"

    return jsonify({
        "question": question,
        "type": q_type,
        "thought": thought,
        "answer": answer
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
