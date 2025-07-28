import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载训练好的模型和分词器
tokenizer = BertTokenizer.from_pretrained("macbert_classifier")
model = BertForSequenceClassification.from_pretrained("macbert_classifier")

def predict(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "简单问题" if prediction == 0 else "复杂问题"

if __name__ == "__main__":
    print("\n进入人工测试模式，输入问题（输入 exit 退出）：")
    while True:
        user_input = input("请输入问题：")
        if user_input.strip().lower() == "exit":
            print("已退出人工测试。")
            break
        result = predict(user_input)
        print(f"模型判断：{result}\n") 