import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification

# 加载训练好的模型和分词器
tokenizer = BertTokenizer.from_pretrained("macbert_classifier")
model = BertForSequenceClassification.from_pretrained("macbert_classifier")

# 测试函数
def predict(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "简单问题" if prediction == 0 else "复杂问题"

# 加载测试问题
with open("dataset/test_questions.json", 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 统计准确度
correct = 0
total = len(test_data)

print(f"开始测试共 {total} 个问题...\n")

# 测试所有问题
for idx, item in enumerate(test_data, start=1):
    question = item["question"]
    true_label = "简单问题" if item["label"] == 0 else "复杂问题"
    predicted = predict(question)
    
    is_correct = (true_label == predicted)
    if is_correct:
        correct += 1

    print(f"[{idx}] 问题: {question}")
    print(f"     真实类别: {true_label}")
    print(f"     预测类别: {predicted}")
    print(f"     预测结果: {'✅ 正确' if is_correct else '❌ 错误'}")
    print("-" * 50)

# 打印准确率
accuracy = correct / total * 100
print(f"\n测试完成！准确率: {accuracy:.2f}% ({correct}/{total})")
