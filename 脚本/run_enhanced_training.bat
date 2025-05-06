@echo off
call conda activate qa_classifier
python train_with_generated.py
echo 增强模型训练完成！ 