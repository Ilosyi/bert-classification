import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification

# 加载增强训练的模型和分词器
tokenizer = BertTokenizer.from_pretrained("macbert_classifier_enhanced")
model = BertForSequenceClassification.from_pretrained("macbert_classifier_enhanced")

# 测试函数
def predict(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    if prediction == 0:
        return "简单问题"
    else:
        return "复杂问题"

# 加载测试问题 - 使用生成的测试数据
with open("dataset/generated_test_data.json", 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 统计准确度
correct = 0
total = len(test_data)

print(f"开始测试增强模型共 {total} 个问题...\n")

# 测试所有问题
for item in test_data:
    question = item["question"]
    true_label = "简单问题" if item["label"] == 0 else "复杂问题"
    predicted = predict(question)
    
    is_correct = true_label == predicted
    if is_correct:
        correct += 1
    
    print(f"问题: {question}")
    print(f"真实类别: {true_label}")
    print(f"预测类别: {predicted}")
    print(f"预测{'正确' if is_correct else '错误'}")
    print("-" * 30)

# 打印准确率
accuracy = correct / total * 100
print(f"\n测试完成！增强模型准确率: {accuracy:.2f}% ({correct}/{total})") 