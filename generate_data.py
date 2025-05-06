import json
import random

# 简单问题模板
simple_templates = [
    "{}是什么？",
    "{}的定义是什么？",
    "{}的特点是什么？",
    "{}在哪里？",
    "{}有多少个？",
    "{}的历史是什么？",
    "{}的作用是什么？",
    "{}是谁发明的？",
    "{}是哪一年开始的？",
    "{}的类型有哪些？"
]

# 复杂问题模板
complex_templates = [
    "如何评价{}对{}的影响？",
    "分析{}与{}的关系。",
    "{}对{}的作用机制是什么？",
    "为什么{}会导致{}？",
    "比较{}和{}的异同。",
    "论述{}在{}中的应用。",
    "{}如何改变{}的发展？",
    "从{}角度分析{}问题。",
    "评价{}在解决{}问题中的效果。",
    "{}与{}之间存在什么矛盾？如何解决？"
]

# 简单问题主题
simple_topics = [
    "太阳系", "电子", "植物光合作用", "动物细胞", "恐龙", "火山", "地震", 
    "海洋", "森林", "城市", "货币", "电脑", "手机", "互联网", "元素周期表",
    "历史事件", "文艺复兴", "工业革命", "中国古代四大发明", "DNA", "维生素",
    "心脏", "大脑", "肌肉", "骨骼", "疫苗", "抗生素", "手术", "医院", "学校"
]

# 复杂问题主题 (主题A, 主题B)
complex_topics = [
    ("人工智能", "就业市场"),
    ("基因编辑", "伦理问题"),
    ("全球化", "文化多样性"),
    ("社交媒体", "心理健康"),
    ("机器学习", "数据隐私"),
    ("自动驾驶", "交通安全"),
    ("区块链", "金融体系"),
    ("可再生能源", "环境保护"),
    ("大数据", "决策过程"),
    ("虚拟现实", "教育方法"),
    ("太空探索", "资源分配"),
    ("生物技术", "医疗发展"),
    ("核能", "能源安全"),
    ("人口老龄化", "社会保障"),
    ("城市化", "生态环境")
]

def generate_simple_questions(count):
    """生成简单问题"""
    questions = []
    for _ in range(count):
        template = random.choice(simple_templates)
        topic = random.choice(simple_topics)
        question = template.format(topic)
        questions.append({"question": question, "label": 0})
    return questions

def generate_complex_questions(count):
    """生成复杂问题"""
    questions = []
    for _ in range(count):
        template = random.choice(complex_templates)
        topic_a, topic_b = random.choice(complex_topics)
        question = template.format(topic_a, topic_b)
        questions.append({"question": question, "label": 1})
    return questions

def main():
    # 生成训练数据
    train_simple = generate_simple_questions(30)
    train_complex = generate_complex_questions(30)
    train_data = train_simple + train_complex
    random.shuffle(train_data)
    
    # 生成测试数据
    test_simple = generate_simple_questions(10)
    test_complex = generate_complex_questions(10)
    test_data = test_simple + test_complex
    random.shuffle(test_data)
    
    # 保存训练数据
    with open("dataset/generated_train_data.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存测试数据
    with open("dataset/generated_test_data.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"已生成 {len(train_data)} 个训练问题，保存至 dataset/generated_train_data.json")
    print(f"已生成 {len(test_data)} 个测试问题，保存至 dataset/generated_test_data.json")

if __name__ == "__main__":
    main() 