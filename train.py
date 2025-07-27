import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 自定义数据集类，用于加载和处理数据
class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=64):
        # 从JSON文件加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self.samples = raw  # 存储原始数据
        self.tokenizer = tokenizer  # 分词器
        self.max_length = max_length  # 输入序列的最大长度

    def __len__(self):
        return len(self.samples)  # 返回数据集的大小

    def __getitem__(self, idx):
        item = self.samples[idx]  # 获取第idx个样本
        # 使用分词器对文本进行编码
        enc = self.tokenizer(
            item['question'],  # 输入文本
            truncation=True,  # 截断超过max_length的文本
            padding='max_length',  # 填充到max_length
            max_length=self.max_length,  # 最大长度
            return_tensors='pt'  # 返回PyTorch张量
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),  # 输入ID（去除多余的维度）
            'attention_mask': enc['attention_mask'].squeeze(),  # 注意力掩码
            'labels': torch.tensor(item['label'], dtype=torch.long)  # 标签
        }

def main():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 从Hugging Face下载模型
    model_name = "hfl/chinese-macbert-base"  # 模型名称
    try:
        # 尝试在线加载分词器和模型
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=False)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except Exception as e:
        print(f"无法从在线仓库加载模型: {e}")
        print("尝试使用离线缓存...")
        # 如果在线加载失败，尝试从本地缓存加载
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, local_files_only=True)
    
    # 将模型移动到GPU（如果可用）
    model.to(device)
    
    # 加载数据集
    dataset = QADataset("dataset/simple_vs_complex_cn.json", tokenizer)
    # 划分训练集和验证集（80%训练，20%验证）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 训练参数配置
    training_args = TrainingArguments(
        output_dir="./results",  # 输出目录
        num_train_epochs=4,  # 训练轮数
        per_device_train_batch_size=2,  # 每个设备的训练批次大小
        per_device_eval_batch_size=2,  # 每个设备的验证批次大小
        warmup_steps=10,  # 预热步数
        evaluation_strategy="epoch",  # 每个epoch结束后评估
        logging_dir="./logs",  # 日志目录
        logging_steps=10,  # 每10步记录一次日志
        save_strategy="epoch",  # 每个epoch结束后保存模型
        no_cuda=False,  # 不禁用CUDA
        fp16=False,  # 不启用半精度训练
        dataloader_num_workers=2  # 数据加载的并行进程数
    )

    # 创建Trainer实例
    trainer = Trainer(
        model=model,  # 模型
        args=training_args,  # 训练参数
        train_dataset=train_dataset,  # 训练集
        eval_dataset=val_dataset  # 验证集
    )

    print("开始训练...")
    trainer.train()  # 开始训练
    
    # 保存模型和分词器
    model.save_pretrained("macbert_classifier")
    tokenizer.save_pretrained("macbert_classifier")
    print("训练完成，模型已保存至 macbert_classifier 目录")

if __name__ == "__main__":
    main()
