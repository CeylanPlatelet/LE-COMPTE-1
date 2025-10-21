#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATLAB工具模块 - 提供Python调用MATLAB的功能
"""

import os
import json
import tempfile
import subprocess
import logging

# 配置日志
logger = logging.getLogger(__name__)

def run_matlab_script(script_name, input_params):
    """
    运行MATLAB脚本并返回结果
    
    Args:
        script_name (str): MATLAB脚本名称，如'Main_program_Q'或'Main_program_T'
        input_params (dict): 输入参数，包含height, width, volume, material等
    
    Returns:
        dict: MATLAB脚本的输出结果
    """
    # 创建临时文件来存储输入参数
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_input_path = temp_file.name
        json.dump(input_params, temp_file)
    
    # 创建临时文件来存储输出结果
    temp_output_path = tempfile.mktemp(suffix='.json')
    
    # 获取修改后副本目录的绝对路径
    matlab_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        '..',
        '修改后副本'
    ))
    
    # 构建MATLAB命令
    matlab_cmd = [
        'matlab',
        '-nodisplay',
        '-nosplash',
        '-nodesktop',
        '-r',
        f"try; addpath('{matlab_dir}'); "
        f"input_data = jsondecode(fileread('{temp_input_path}')); "
        f"input1 = input_data.height; "
        f"input2 = input_data.width; "
        f"input3 = input_data.volume; "
        f"input4 = input_data.material_value; "
        f"result = {script_name}(input1, input2, input3, input4); "
        f"fileID = fopen('{temp_output_path}', 'w'); "
        f"fprintf(fileID, '%f', result); "
        f"fclose(fileID); "
        f"exit; catch e; disp(getReport(e)); exit(1); end;"
    ]
    
    try:
        # 运行MATLAB命令
        logger.info(f"Running MATLAB script: {script_name}")
        process = subprocess.Popen(
            matlab_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=60)  # 设置超时时间为60秒
        
        # 检查MATLAB是否成功运行
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8')
            logger.error(f"MATLAB execution error: {error_msg}")
            return {"error": "MATLAB执行错误", "details": error_msg}
        
        # 读取结果
        with open(temp_output_path, 'r') as f:
            result = float(f.read().strip())
        
        logger.info(f"MATLAB script {script_name} executed successfully, result: {result}")
        return {"result": result}
    
    except subprocess.TimeoutExpired:
        process.kill()
        logger.error(f"MATLAB script {script_name} execution timeout")
        return {"error": "MATLAB执行超时"}
    except Exception as e:
        logger.error(f"Error running MATLAB script {script_name}: {e}")
        return {"error": str(e)}
    finally:
        # 清理临时文件
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if os.path.exists(temp_output_path):
            os.unlink(temp_output_path)