#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本 - 启动后端应用
"""

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入后端应用
from backend.app import create_app

if __name__ == '__main__':
    # 创建应用
    app = create_app()
    
    # 启动应用
    print("Starting Dam Monitoring System Web Server...")
    print("Open your browser and go to: http://localhost:5080")
    app.run(debug=True, host='0.0.0.0', port=5080)