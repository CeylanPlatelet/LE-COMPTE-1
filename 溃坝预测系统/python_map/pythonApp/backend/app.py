#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后端主应用模块 - 创建Flask应用并注册路由
"""

import os
import sys
import logging
from flask import Flask, send_from_directory, render_template
from flask_cors import CORS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """
    创建Flask应用
    
    Returns:
        Flask: Flask应用实例
    """
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # 创建Flask应用
    app = Flask(
        __name__,
        static_folder=os.path.join(parent_dir, 'frontend'),
        template_folder=os.path.join(parent_dir, 'frontend')
    )
    
    # 配置跨域
    CORS(app)
    
    # 注册API路由
    from backend.api.api_routes import api_bp
    app.register_blueprint(api_bp)
    
    # 前端路由
    @app.route('/')
    def index():
        """首页"""
        return send_from_directory(app.static_folder, 'index.html')
    
    @app.route('/<path:path>')
    def static_files(path):
        """静态文件"""
        return send_from_directory(app.static_folder, path)
    
    @app.route('/pages/<path:filename>')
    def serve_pages(filename):
        """页面文件"""
        return send_from_directory(os.path.join(app.static_folder, 'pages'), filename)
    
    @app.errorhandler(404)
    def page_not_found(e):
        """404错误处理"""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """500错误处理"""
        return render_template('500.html'), 500
    
    return app

if __name__ == '__main__':
    # 创建应用
    app = create_app()
    
    # 启动应用
    print("Starting Dam Monitoring System Web Server...")
    print("Open your browser and go to: http://localhost:5080")
    app.run(debug=True, host='0.0.0.0', port=5080)