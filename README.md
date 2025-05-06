# 简单与复杂问题分类器

这个项目使用中文MacBERT预训练模型进行简单和复杂问题的二分类任务。

## 项目结构

```
.
├── dataset/                        # 数据集目录
│   ├── simple_vs_complex_cn.json   # 原始训练数据集
│   ├── test_questions.json         # 手动创建的测试集
│   ├── generated_train_data.json   # 自动生成的训练数据（运行后生成）
│   ├── generated_test_data.json    # 自动生成的测试数据（运行后生成）
│   └── combined_train_data.json    # 合并的训练数据（运行后生成）
├── 脚本/ 
   ├── setup_env.bat                   # 环境设置批处理文件
   ├── run_training.bat                # 运行训练的批处理文件
   ├── run_test.bat                    # 运行测试的批处理文件
   ├── run_generate_data.bat           # 运行数据生成的批处理文件
   ├── run_enhanced_training.bat       # 运行增强训练的批处理文件
   ├── run_test_enhanced.bat           # 运行增强模型测试的批处理文件
   ├── run_download_model.bat          # 运行模型下载的批处理文件
   ├── run_local_model.bat             # 运行本地模型训练的批处理文件
   └── run_random_model.bat            # 运行随机模型训练的批处理文件
├── macbert_classifier/             #训练好的模型
├── macbert_classifier_enhanced/    #使用自动生成训练集generated_train_data.json训练好的模型
├── train.py                        # 基础训练脚本
├── test.py                         # 测试脚本
├── generate_data.py                # 数据生成脚本
├── train_with_generated.py         # 使用扩充数据集的训练脚本
├── test_enhanced.py                # 测试增强模型的脚本
```

## 环境设置

### 使用Conda

1. 创建新的conda环境：
   ```bash
   conda create -n qa_classifier python=3.8 -y
   ```

2. 激活环境：
   ```bash
   conda activate qa_classifier
   ```

3. 安装依赖包(建议安装GPU版本的torch)：
   ```bash
   pip install torch transformers datasets
   ```

### 使用批处理文件（Windows）

运行提供的批处理文件设置环境：
```
.\setup_env.bat
```

## 数据集

- **simple_vs_complex_cn.json**: 原始训练数据集，包含简单问题（标签0）和复杂问题（标签1）
- **test_questions.json**: 手动创建的测试集
- 可以使用 `generate_data.py` 生成更多训练和测试数据
- 可以使用大模型生成更多训练和测试数据

## 模型下载问题解决方案

如果您遇到无法下载预训练模型的问题，可以尝试以下几种解决方案：
### 1. 使用Clash系统代理
在CMD中输入以下命令
```
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
```
之后再用python运行相应代码，7890是Clash的默认端口，你也可以更改，注意要打开局域网连接

### 2. 尝试预先下载模型

运行模型下载脚本，它会尝试多种方式下载预训练模型：
```
.\run_download_model.bat
```

此脚本会尝试：
- 使用清华镜像下载
- 清除代理设置
- 尝试多个不同的预训练模型



## 训练模型

### 基础训练

```bash
python train.py
```
或使用批处理文件：
```
.\run_training.bat
```

### 使用扩充数据集训练

1. 首先生成更多数据：
   ```bash
   python generate_data.py
   ```
   或使用批处理文件：
   ```
   .\run_generate_data.bat
   ```

2. 然后使用扩充数据集训练：
   ```bash
   python train_with_generated.py
   ```
   或使用批处理文件：
   ```
   .\run_enhanced_training.bat
   ```

## GPU训练

训练脚本已配置为自动检测并使用可用的GPU。以下是GPU相关的配置：

1. 检测GPU是否可用：
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. 将模型移至GPU设备：
   ```python
   model.to(device)
   ```

3. 训练参数设置：
   ```python
   training_args = TrainingArguments(
       # 其他参数...
       no_cuda=False,  # 设为True会禁用CUDA
       fp16=False,     # 若GPU支持，可设为True启用半精度训练
       dataloader_num_workers=2  # 数据加载并行进程数
   )
   ```

### 加速训练的提示

- 如果有高性能GPU，可以增加批量大小（batch size）
- 启用混合精度训练（fp16=True）可以减少内存使用并加速训练
- 可以根据需要调整`num_train_epochs`参数

## 测试模型

### 测试基础模型

```bash
python test.py
```
或使用批处理文件：
```
.\run_test.bat
```

### 测试增强模型

```bash
python test_enhanced.py
```
或使用批处理文件：
```
.\run_test_enhanced.bat
```

## 常见问题

### 模型下载问题

如果遇到模型下载问题：

1. 清除代理设置：
   ```python
   import os
   os.environ['HTTP_PROXY'] = ''
   os.environ['HTTPS_PROXY'] = ''
   ```

2. 使用镜像站点：
   ```python
   tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", mirror='tuna')
   ```

3. 尝试其他模型，如`bert-base-chinese`而不是`hfl/chinese-macbert-base`

4. 使用随机初始化的模型完全绕过下载需求

### GPU内存不足

如果遇到GPU内存不足的问题，可以：

1. 减小批量大小（per_device_train_batch_size）
2. 启用梯度累积（gradient_accumulation_steps）
3. 使用混合精度训练（fp16=True）
4. 减少模型层数（如使用6层而不是12层）

## 模型效果

该模型能够区分简单和复杂问题，简单问题通常是事实性问题，而复杂问题通常需要分析、评估或解释。 