# 配置文件
import os

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-90b9bfa6d6b94c20874f04b6f55b315c')  # 从环境变量读取，如果没有则使用默认值
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# 本地模型配置
LOCAL_MODEL_URL = "http://127.0.0.1:8080/completion"

# Flask 服务器配置
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# 模型参数配置
DEEPSEEK_MODEL = "deepseek-chat"  # 或 "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.7
DEEPSEEK_MAX_TOKENS = 1000
LOCAL_MODEL_MAX_TOKENS = 256

# 分类模型配置
CLASSIFIER_MAX_LENGTH = 64 