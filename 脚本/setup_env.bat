@echo off
call conda activate qa_classifier || call conda create -n qa_classifier python=3.8 -y && call conda activate qa_classifier
pip install torch transformers datasets
echo 环境设置完成！ 