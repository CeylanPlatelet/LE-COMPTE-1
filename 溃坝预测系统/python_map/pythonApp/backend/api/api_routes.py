#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API路由模块 - 处理所有API请求
"""

import json
import os
import sys
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app

# 导入工具函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.matlab_utils import run_matlab_script
from utils.model_utils import run_python_model
from utils.data_utils import get_case_data, get_dam_data, get_locations

# 创建Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/cases', methods=['GET'])
def get_cases():
    """获取所有案例数据"""
    try:
        cases = get_case_data()
        return jsonify({'cases': cases})
    except Exception as e:
        current_app.logger.error(f"Error getting cases: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/cases/<int:case_id>', methods=['GET'])
def get_case(case_id):
    """获取指定案例数据"""
    try:
        cases = get_case_data()
        case = next((c for c in cases if c['id'] == case_id), None)
        if case:
            return jsonify(case)
        return jsonify({'error': 'Case not found'}), 404
    except Exception as e:
        current_app.logger.error(f"Error getting case {case_id}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/cases/<int:case_id>/dam-data', methods=['GET'])
def get_dam_data_api(case_id):
    """获取指定案例的大坝数据"""
    try:
        dam_data = get_dam_data(case_id)
        return jsonify(dam_data)
    except Exception as e:
        current_app.logger.error(f"Error getting dam data for case {case_id}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/locations', methods=['GET'])
def get_locations_api():
    """获取所有地点数据"""
    try:
        locations = get_locations()
        return jsonify({'locations': locations})
    except Exception as e:
        current_app.logger.error(f"Error getting locations: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models/calculate', methods=['POST'])
def calculate_model():
    """计算模型结果"""
    try:
        data = request.get_json()
        model_type = data.get('model', 'model1')
        parameters = data.get('parameters', {})
        
        # 参数验证
        required_params = ['height', 'width', 'volume', 'material']
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            return jsonify({
                'error': '缺少必要参数',
                'missing_parameters': missing_params
            }), 400
        
        # 提取参数
        height = float(parameters.get('height', 0))
        width = float(parameters.get('width', 0))
        volume = float(parameters.get('volume', 0))
        material = parameters.get('material', '土质为主')
        
        # 根据材料类型映射到数值
        material_map = {
            '土质为主': 16894.78, 
            '土含块石': 8211.76, 
            '块石夹土': 3017.18, 
            '块石为主': 3157.55
        }
        material_value = material_map.get(material, 16894.78)
        
        # 准备输入参数
        input_params = {
            'height': height,
            'width': width,
            'volume': volume,
            'material': material,
            'material_value': material_value
        }
        
        # 根据模型类型调用不同的计算函数
        if model_type == 'peak_flow':
            # 调用MATLAB脚本计算峰值流量
            result = run_matlab_script('Main_program_Q', input_params)
        elif model_type == 'breach_time':
            # 调用MATLAB脚本计算溃决历时
            result = run_matlab_script('Main_program_T', input_params)
        elif model_type in ['VANRIJN', 'BREACH', 'CHEN', 'DAMBRK', 'MPM', 'SMART', 'WANG', 'YANG']:
            # 调用Python模型
            result = run_python_model(model_type, input_params)
        else:
            # 未知模型类型
            return jsonify({
                'error': '未知模型类型',
                'model': model_type
            }), 400
        
        # 检查计算结果是否有错误
        if 'error' in result:
            return jsonify({
                'error': '计算错误',
                'details': result['error']
            }), 500
        
        # 构建响应
        response = {
            'model': model_type,
            'parameters': parameters,
            'result': result['result']
        }
        
        # 如果是Python模型，添加图表数据
        if model_type in ['VANRIJN', 'BREACH', 'CHEN', 'DAMBRK', 'MPM', 'SMART', 'WANG', 'YANG']:
            response['chart_data'] = result.get('chart_data', {})
        
        return jsonify(response)
    
    except ValueError as e:
        current_app.logger.error(f"Parameter format error: {e}")
        return jsonify({
            'error': '参数格式错误',
            'details': str(e)
        }), 400
    except Exception as e:
        current_app.logger.error(f"Server error: {e}")
        return jsonify({
            'error': '服务器错误',
            'details': str(e)
        }), 500