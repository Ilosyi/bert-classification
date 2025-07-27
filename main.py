import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import json

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
        "n_predict": 128  # 生成最大token数
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", "[本地模型无返回内容]")
    except Exception as e:
        return f"[本地模型调用失败: {e}]"

def main():
    print("欢迎使用智能问答系统，输入问题（输入exit退出）：")
    while True:
        question = input("请输入问题：").strip()
        if question.lower() == "exit":
            print("已退出。")
            break
        q_type = classify_question(question)
        print(f"问题类型：{q_type}")
        if q_type == "简单问题":
            answer = call_llama_local(question)
            print(f"本地大模型回答：{answer}")
        else:
            print("[info] 复杂问题将由云端模型处理（当前未部署）")

if __name__ == "__main__":
    main() python