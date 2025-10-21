#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工具模块 - 提供Python模型的调用功能
"""

import os
import sys
import logging
import importlib
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)

def run_python_model(model_name, parameters):
    """
    运行Python模型并返回结果
    
    Args:
        model_name (str): 模型名称，如'VANRIJN'、'BREACH'等
        parameters (dict): 模型参数
    
    Returns:
        dict: 模型计算结果
    """
    try:
        # 添加模型目录到Python路径
        model_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if model_dir not in sys.path:
            sys.path.append(model_dir)
        
        logger.info(f"Running Python model: {model_name}")
        
        # 动态导入模型模块
        try:
            model_module = importlib.import_module(model_name)
        except ImportError:
            logger.error(f"Failed to import model module: {model_name}")
            return {"error": f"模型 {model_name} 不存在或无法导入"}
        
        # 获取模型函数
        model_func = getattr(model_module, model_name, None)
        if not model_func:
            logger.error(f"Model function {model_name} not found in module {model_name}")
            return {"error": f"模型函数 {model_name} 在模块中不存在"}
        
        # 运行模型函数
        result = model_func()
        
        # 提取结果
        if isinstance(result, (int, float)):
            # 如果结果是数值，直接返回
            return {"result": result}
        elif isinstance(result, dict):
            # 如果结果是字典，直接返回
            return result
        elif isinstance(result, (list, np.ndarray)):
            # 如果结果是列表或数组，提取最大值作为峰值
            max_value = max(result) if len(result) > 0 else 0
            
            # 创建图表数据
            x = list(range(len(result)))
            y = [float(val) for val in result]
            
            return {
                "result": max_value,
                "chart_data": {
                    "x": x,
                    "y": y
                }
            }
        else:
            # 其他类型，尝试转换为浮点数
            try:
                return {"result": float(result)}
            except (ValueError, TypeError):
                logger.error(f"Unable to convert model result to float: {result}")
                return {"error": "模型结果无法转换为数值"}
    
    except Exception as e:
        logger.error(f"Error running Python model {model_name}: {e}")
        return {"error": str(e)}