#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据工具模块 - 提供数据处理和获取功能
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)

# 配置路径
INPUT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "myFloodRoutingSys" / "dam" / "input"
INPUT_FILE = "params.xlsx"

def get_case_data():
    """
    获取案例数据
    
    Returns:
        list: 案例数据列表
    """
    try:
        if INPUT_FILE and (INPUT_DIR / INPUT_FILE).exists():
            # 加载数据
            df = pd.read_excel(INPUT_DIR / INPUT_FILE, sheet_name=None)
            
            # 处理不同的表
            case_data = []
            for sheet_name, sheet_df in df.items():
                if '案例' in sheet_name or 'case' in sheet_name.lower():
                    for _, row in sheet_df.iterrows():
                        case = {
                            'id': len(case_data) + 1,
                            'name': str(row.get('名称', f'案例{len(case_data) + 1}')),
                            'longitude': float(row.get('经度', 100.0 + np.random.randn())),
                            'latitude': float(row.get('纬度', 30.0 + np.random.randn())),
                            'slope': float(row.get('坡度', 1)),
                            'slopeLength': float(row.get('坡面长', 1)),
                            'town': float(row.get('城镇', 11)),
                            'date': float(row.get('发生日', 1)),
                            'width': float(row.get('坡宽', 1)),
                            'drainageLength': float(row.get('汇水长', 11)),
                            'townPoint': float(row.get('城镇点', 1)),
                            'basinE': float(row.get('流域E', 1)),
                            'slopeLength2': float(row.get('坡长', 1)),
                            'upstream': float(row.get('上游', 11)),
                            'townPeople': float(row.get('城镇人', 1)),
                            'caseI': float(row.get('案例I', 1)),
                            'vegetationI': float(row.get('植被I', 1)),
                            'overflow': float(row.get('漫流', 11)),
                            'economy': float(row.get('经济', 11))
                        }
                        case_data.append(case)
            
            if not case_data:
                # 创建示例数据
                return create_sample_data()
            
            return case_data
        else:
            return create_sample_data()
    
    except Exception as e:
        logger.error(f"Error loading case data: {e}")
        return create_sample_data()

def create_sample_data():
    """
    创建示例数据
    
    Returns:
        list: 示例数据列表
    """
    return [
        {
            'id': 1,
            'name': '丹巴梅龙沟',
            'longitude': 102.0250,
            'latitude': 30.9810,
            'risk': 'high',
            'date': '2008-05-12',
            'height': 85,
            'width': 320,
            'volume': 680,
            'material': '土含块石'
        },
        {
            'id': 2,
            'name': '泸定县烂田湾',
            'longitude': 102.1797,
            'latitude': 29.5909,
            'risk': 'medium',
            'date': '2010-08-13',
            'height': 65,
            'width': 280,
            'volume': 450,
            'material': '块石夹土'
        },
        {
            'id': 3,
            'name': '唐家山堰塞坝',
            'longitude': 104.4272,
            'latitude': 31.8467,
            'risk': 'high',
            'date': '2008-05-12',
            'height': 124,
            'width': 803,
            'volume': 2400,
            'material': '块石为主'
        }
    ]

def get_dam_data(case_id):
    """
    获取大坝数据
    
    Args:
        case_id (int): 案例ID
    
    Returns:
        dict: 大坝数据
    """
    try:
        cases = get_case_data()
        case = next((c for c in cases if c['id'] == case_id), None)
        
        if not case:
            return {'error': 'Case not found'}
        
        # 生成模拟数据
        time_series = np.linspace(0, 10, 20)
        water_level = 70 + 5 * np.sin(time_series) + np.random.randn(20)
        flow_rate = 100 + 20 * np.sin(time_series + 1) + np.random.randn(20) * 5
        
        return {
            'case_id': case_id,
            'name': case.get('name', f'案例{case_id}'),
            'time_series': time_series.tolist(),
            'water_level': water_level.tolist(),
            'flow_rate': flow_rate.tolist()
        }
    
    except Exception as e:
        logger.error(f"Error getting dam data for case {case_id}: {e}")
        return {'error': str(e)}

def get_locations():
    """
    获取所有地点数据
    
    Returns:
        list: 地点数据列表
    """
    try:
        cases = get_case_data()
        locations = []
        
        for case in cases:
            locations.append({
                'Name': case.get('name', ''),
                'Lat': case.get('latitude', 0),
                'Lon': case.get('longitude', 0),
                'Risk': case.get('risk', 'medium')
            })
        
        return locations
    
    except Exception as e:
        logger.error(f"Error getting locations: {e}")
        return [
            {'Name': '丹巴梅龙沟', 'Lat': 30.9810, 'Lon': 102.0250, 'Risk': 'high'},
            {'Name': '泸定县烂田湾', 'Lat': 29.5909, 'Lon': 102.1797, 'Risk': 'medium'},
            {'Name': '唐家山堰塞坝', 'Lat': 31.8467, 'Lon': 104.4272, 'Risk': 'high'}
        ]