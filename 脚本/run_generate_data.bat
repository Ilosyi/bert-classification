@echo off
call conda activate qa_classifier
python generate_data.py
echo 数据生成完成！ 