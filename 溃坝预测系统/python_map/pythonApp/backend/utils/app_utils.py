#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用工具模块 - 从APP1.py中提取的有用功能
"""

import os
import re
import sys
import json
import csv
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression

# 配置路径
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    INPUT_DIR = PROJECT_ROOT / "myFloodRoutingSys" / "dam" / "input"
    OUTPUT_DIR = PROJECT_ROOT / "myFloodRoutingSys" / "dam" / "output"
    INPUT_FILE = INPUT_DIR / "params3.xlsx"
except Exception as e:
    print(f"Error configuring paths: {e}")
    PROJECT_ROOT = Path.cwd()
    INPUT_DIR = PROJECT_ROOT / "myFloodRoutingSys" / "dam" / "input"
    OUTPUT_DIR = PROJECT_ROOT / "myFloodRoutingSys" / "dam" / "output"
    INPUT_FILE = INPUT_DIR / "params3.xlsx"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _find_sheet_name(xlsx, target_name, fallback_index=None, require_lonlat=False):
    """
    在工作簿中查找与 target_name 最匹配的工作表名：
    - 忽略大小写、空格、全角括号、下划线等
    - target_name='Sheet3' 时容忍 'sheet 3'、'Sheet_3'、'工作表3' 等
    - require_lonlat=True 时，会优先返回含经纬度列的表
    """
    try:
        xl = pd.ExcelFile(xlsx)
        names = xl.sheet_names
    except Exception as e:
        print(f"[DATA] open excel failed: {e}")
        return target_name

    def norm(s):
        return re.sub(r'[\s\u3000()（）_-]', '', str(s).strip().lower())

    want = norm(target_name)

    # 1) 严格归一匹配
    for n in names:
        if norm(n) == want:
            return n

    # 2) "Sheet3 / 工作表3" 类似的松匹配
    if want == 'sheet3':
        for n in names:
            if re.search(r'(sheet|工作表)?0*3$', norm(n)):
                return n

    # 3) 需要经纬度列时，找一个包含经纬度列的表
    if require_lonlat:
        for n in names:
            try:
                df = pd.read_excel(xlsx, sheet_name=n)
                cols = [c.lower() for c in df.columns]
                if any('lon' in c for c in cols) and any('lat' in c for c in cols):
                    return n
            except:
                pass

    # 4) 返回第一个表或指定索引的表
    if fallback_index is not None and 0 <= fallback_index < len(names):
        return names[fallback_index]
    elif names:
        return names[0]
    else:
        return target_name

def _generate_demo_series(kind: str, n: int = 200):
    """生成演示数据序列"""
    x = np.linspace(0, 10, n)
    
    if kind == 'sin':
        y = np.sin(x) + np.random.normal(0, 0.1, n)
    elif kind == 'cos':
        y = np.cos(x) + np.random.normal(0, 0.1, n)
    elif kind == 'tan':
        y = np.tan(x/3) + np.random.normal(0, 0.1, n)
    elif kind == 'exp':
        y = np.exp(x/3) / 100 + np.random.normal(0, 0.1, n)
    elif kind == 'log':
        y = np.log(x + 0.1) + np.random.normal(0, 0.1, n)
    elif kind == 'linear':
        y = 0.5 * x + 1 + np.random.normal(0, 0.1, n)
    else:
        y = np.random.normal(0, 1, n)
    
    return x, y

def create_map_figure():
    """创建地图图形"""
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # 绘制中国地图轮廓（简化版）
    # 这里只是一个示例，实际应用中应该使用真实的地图数据
    x = [100, 110, 120, 130, 140, 130, 120, 110, 100, 90, 80, 90, 100]
    y = [20, 25, 30, 35, 40, 45, 50, 45, 40, 35, 30, 25, 20]
    ax.plot(x, y, 'b-', linewidth=2)
    
    # 添加一些城市标记
    cities = {
        '北京': (116.4, 39.9),
        '上海': (121.5, 31.2),
        '广州': (113.3, 23.1),
        '成都': (104.1, 30.7),
        '西安': (108.9, 34.3)
    }
    
    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, 'ro', markersize=5)
        ax.text(lon, lat, city, fontsize=8)
    
    ax.set_title('中国地图')
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.grid(True)
    
    return fig

def _is_nan(x):
    """检查值是否为NaN"""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and not x.strip():
        return True
    return False

def _best_match(target, candidates):
    """找到最佳匹配"""
    if not target or not candidates:
        return None
    
    matches = difflib.get_close_matches(target, candidates, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    
    # 尝试部分匹配
    for c in candidates:
        if target.lower() in c.lower() or c.lower() in target.lower():
            return c
    
    return None

def _find_input_file(input_dir, input_file):
    """查找输入文件"""
    if isinstance(input_file, (str, Path)):
        p = Path(input_file)
        if p.exists():
            return p
    
    if isinstance(input_dir, (str, Path)):
        d = Path(input_dir)
        if d.exists() and d.is_dir():
            for f in d.glob("*.xlsx"):
                return f
    
    return None

def _fill_case_info_from_sheet1(input_dir=None, input_file=None, max_fields=20):
    """从Sheet1填充案例信息"""
    if input_dir is None:
        input_dir = INPUT_DIR
    if input_file is None:
        input_file = INPUT_FILE
    
    file_path = _find_input_file(input_dir, input_file)
    if not file_path:
        print(f"[DATA] Input file not found: {input_file}")
        return []
    
    try:
        sheet_name = _find_sheet_name(file_path, "Sheet1", fallback_index=0)
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 查找包含经纬度的列
        cols = [str(c).lower() for c in df.columns]
        lon_col = next((c for i, c in enumerate(df.columns) if '经度' in str(c) or 'lon' in str(c).lower()), None)
        lat_col = next((c for i, c in enumerate(df.columns) if '纬度' in str(c) or 'lat' in str(c).lower()), None)
        
        if not lon_col or not lat_col:
            print(f"[DATA] Longitude/Latitude columns not found in {sheet_name}")
            return []
        
        cases = []
        for _, row in df.iterrows():
            case = {'id': len(cases) + 1}
            
            # 添加经纬度
            lon = row.get(lon_col)
            lat = row.get(lat_col)
            if not _is_nan(lon) and not _is_nan(lat):
                case['longitude'] = float(lon)
                case['latitude'] = float(lat)
            
            # 添加其他字段
            for col in df.columns[:max_fields]:
                if col != lon_col and col != lat_col and not _is_nan(row.get(col)):
                    case[str(col)] = row.get(col)
            
            cases.append(case)
        
        return cases
    
    except Exception as e:
        print(f"[DATA] Error reading case data: {e}")
        return []

def _update_region_overview_and_map(input_dir=None, input_file=None):
    """更新区域概览和地图"""
    if input_dir is None:
        input_dir = INPUT_DIR
    if input_file is None:
        input_file = INPUT_FILE
    
    file_path = _find_input_file(input_dir, input_file)
    if not file_path:
        print(f"[DATA] Input file not found: {input_file}")
        return None
    
    try:
        # 创建地图图形
        fig = create_map_figure()
        
        # 获取案例数据
        cases = _fill_case_info_from_sheet1(input_dir, input_file)
        
        # 在地图上添加案例标记
        ax = fig.axes[0]
        for case in cases:
            lon = case.get('longitude')
            lat = case.get('latitude')
            if lon and lat:
                ax.plot(lon, lat, 'go', markersize=8)
                ax.text(lon, lat, case.get('name', f"案例{case['id']}"), fontsize=8)
        
        return fig
    
    except Exception as e:
        print(f"[DATA] Error updating map: {e}")
        return None

def fig_to_base64(fig):
    """将matplotlib图形转换为base64编码的字符串"""
    import io
    import base64
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str