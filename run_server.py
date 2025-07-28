#!/usr/bin/env python3
"""
端云协同问答系统启动脚本
"""

import os
import sys
from flask_server import app

def main():
    """主函数"""
    print("=" * 50)
    print("端云协同问答系统")
    print("=" * 50)
    
    # 检查API密钥
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key or api_key == 'sk-你的密钥':
        print("⚠️  警告: 请设置 DeepSeek API 密钥")
        print("方法1: 设置环境变量 DEEPSEEK_API_KEY")
        print("方法2: 在 config.py 中直接修改 DEEPSEEK_API_KEY")
        print()
    
    print("系统配置:")
    print(f"- 本地模型: llama.cpp (端口 8080)")
    print(f"- 云端模型: DeepSeek API")
    print(f"- 分类模型: MacBERT")
    print(f"- Web服务: http://localhost:5000")
    print()
    
    print("启动服务器...")
    print("访问 http://localhost:5000 开始使用")
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    # 启动Flask应用
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main() 