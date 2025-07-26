import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=64):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self.samples = raw
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item['question'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def main():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 从Hugging Face下载模型，添加代理设置和使用本地缓存选项
    model_name = "hfl/chinese-macbert-base"
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=False)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except Exception as e:
        print(f"无法从在线仓库加载模型: {e}")
        print("尝试使用离线缓存...")
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, local_files_only=True)
    
    # 将模型移动到GPU（如果可用）
    model.to(device)
    
    dataset = QADataset("dataset/simple_vs_complex_cn.json", tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=10,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        # GPU相关设置
        no_cuda=False,  # 如果为True则禁用CUDA
        fp16=False,     # 如果GPU支持，可以设为True启用半精度训练
        dataloader_num_workers=2  # 数据加载的并行进程数
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("开始训练...")
    trainer.train()
    
    model.save_pretrained("macbert_classifier")
    tokenizer.save_pretrained("macbert_classifier")
    print("训练完成，模型已保存至 macbert_classifier 目录")

if __name__ == "__main__":
    main()
