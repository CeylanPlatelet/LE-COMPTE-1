#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UND模块 - 集成多个模型并与MATLAB结果进行比较
"""

import numpy as np
import logging

# 配置日志
logger = logging.getLogger(__name__)

def UND():
    """
    运行多个模型并与MATLAB结果进行比较
    
    Returns:
        tuple: 各个模型的结果
    """
    try:
        # 导入MATLAB工具
        from ..utils.matlab_utils import run_matlab_script
        
        # 设置默认参数
        default_params = {
            'height': 85,
            'width': 320,
            'volume': 680,
            'material': '土含块石',
            'material_value': 8211.76
        }
        
        # 运行MATLAB脚本
        output_Q_result = run_matlab_script('Main_program_Q', default_params)
        output_T_result = run_matlab_script('Main_program_T', default_params)
        
        if 'error' in output_Q_result or 'error' in output_T_result:
            logger.error("MATLAB脚本执行失败")
            return None
        
        output_Q = output_Q_result['result']
        output_T = output_T_result['result']
        
        # 导入各个模型
        from . import BREACH, CHEN, DAMBRK, MPM, SMART, VANRIJN, WANG, YANG
        
        # 运行BREACH模型
        T1, Qb1, B1, Z1, Hw1 = BREACH.BREACH()
        T1 = T1.T
        Qb1 = Qb1.T
        B1 = B1.T
        Z1 = Z1.T
        Hw1 = Hw1.T
        max1 = np.argmax(Qb1)
        max_Q1 = Qb1[max1]
        max_T1 = T1[max1]
        
        # 运行CHEN模型
        T2, Qb2, B2, Z2, Hw2 = CHEN.CHEN()
        T2 = T2.T
        Qb2 = Qb2.T
        B2 = B2.T
        Z2 = Z2.T
        Hw2 = Hw2.T
        max2 = np.argmax(Qb2)
        max_Q2 = Qb2[max2]
        max_T2 = T2[max2]
        
        # 运行DAMBRK模型
        T3, Qb3, B3, Z3, Hw3 = DAMBRK.DAMBRK()
        T3 = T3.T
        Qb3 = Qb3.T
        B3 = B3.T
        Z3 = Z3.T
        Hw3 = Hw3.T
        max3 = np.argmax(Qb3)
        max_Q3 = Qb3[max3]
        max_T3 = T3[max3]
        
        # 运行MPM模型
        T4, Qb4, B4, Z4, Hw4 = MPM.MPM()
        T4 = T4.T
        Qb4 = Qb4.T
        B4 = B4.T
        Z4 = Z4.T
        Hw4 = Hw4.T
        max4 = np.argmax(Qb4)
        max_Q4 = Qb4[max4]
        max_T4 = T4[max4]
        
        # 运行SMART模型
        T5, Qb5, B5, Z5, Hw5 = SMART.SMART()
        T5 = T5.T
        Qb5 = Qb5.T
        B5 = B5.T
        Z5 = Z5.T
        Hw5 = Hw5.T
        max5 = np.argmax(Qb5)
        max_Q5 = Qb5[max5]
        max_T5 = T5[max5]
        
        # 运行VANRIJN模型
        T6, Qb6, B6, Z6, Hw6 = VANRIJN.VANRIJN()
        T6 = T6.T
        Qb6 = Qb6.T
        B6 = B6.T
        Z6 = Z6.T
        Hw6 = Hw6.T
        max6 = np.argmax(Qb6)
        max_Q6 = Qb6[max6]
        max_T6 = T6[max6]
        
        # 运行WANG模型
        T7, Qb7, B7, Z7, Hw7 = WANG.WANG()
        T7 = T7.T
        Qb7 = Qb7.T
        B7 = B7.T
        Z7 = Z7.T
        Hw7 = Hw7.T
        max7 = np.argmax(Qb7)
        max_Q7 = Qb7[max7]
        max_T7 = T7[max7]
        
        # 运行YANG模型
        T8, Qb8, B8, Z8, Hw8 = YANG.YANG()
        T8 = T8.T
        Qb8 = Qb8.T
        B8 = B8.T
        Z8 = Z8.T
        Hw8 = Hw8.T
        max8 = np.argmax(Qb8)
        max_Q8 = Qb8[max8]
        max_T8 = T8[max8]
        
        # 比较各个模型的结果
        Q0 = output_Q
        max_Q_values = np.array([max_Q1, max_Q2, max_Q3, max_Q4, max_Q5, max_Q6, max_Q7, max_Q8])
        names = [f"max_Q{i}" for i in range(1, 9)]
        absolute_errors = np.abs(Q0 - max_Q_values)
        ape_values = (absolute_errors / np.abs(Q0)) * 100
        results = sorted(zip(names, ape_values), key=lambda x: x[1])
        best_model_name = results[0][0]  # 最优模型
        best_model_value = results[0][1]  # 最优模型的MAPE
        
        logger.info(f"MATLAB峰值流量预测结果: {output_Q}")
        logger.info(f"MATLAB溃决历时预测结果: {output_T}")
        logger.info(f"最优模型: {best_model_name}")
        logger.info(f"最优模型的MAPE: {best_model_value}")
        
        return {
            'output_Q': output_Q,
            'output_T': output_T,
            'best_model_name': best_model_name,
            'best_model_value': best_model_value,
            'models_data': {
                'BREACH': {'T': T1, 'Q': Qb1, 'B': B1, 'Z': Z1, 'Hw': Hw1, 'max_Q': max_Q1, 'max_T': max_T1},
                'CHEN': {'T': T2, 'Q': Qb2, 'B': B2, 'Z': Z2, 'Hw': Hw2, 'max_Q': max_Q2, 'max_T': max_T2},
                'DAMBRK': {'T': T3, 'Q': Qb3, 'B': B3, 'Z': Z3, 'Hw': Hw3, 'max_Q': max_Q3, 'max_T': max_T3},
                'MPM': {'T': T4, 'Q': Qb4, 'B': B4, 'Z': Z4, 'Hw': Hw4, 'max_Q': max_Q4, 'max_T': max_T4},
                'SMART': {'T': T5, 'Q': Qb5, 'B': B5, 'Z': Z5, 'Hw': Hw5, 'max_Q': max_Q5, 'max_T': max_T5},
                'VANRIJN': {'T': T6, 'Q': Qb6, 'B': B6, 'Z': Z6, 'Hw': Hw6, 'max_Q': max_Q6, 'max_T': max_T6},
                'WANG': {'T': T7, 'Q': Qb7, 'B': B7, 'Z': Z7, 'Hw': Hw7, 'max_Q': max_Q7, 'max_T': max_T7},
                'YANG': {'T': T8, 'Q': Qb8, 'B': B8, 'Z': Z8, 'Hw': Hw8, 'max_Q': max_Q8, 'max_T': max_T8}
            }
        }
    
    except Exception as e:
        logger.error(f"UND模型执行错误: {e}")
        return None