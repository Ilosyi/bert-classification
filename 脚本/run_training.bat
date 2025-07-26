@echo off
call conda activate qa_classifier
python train.py
echo 训练完成！ 