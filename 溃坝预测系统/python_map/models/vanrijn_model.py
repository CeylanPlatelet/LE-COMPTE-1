# -*- coding: utf-8 -*-
"""
Functional wrapper for VANRIJN model.
Returns key time series as a pandas DataFrame instead of plotting.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from config import INPUT_FILE

def run_vanrijn(input_excel: Path = INPUT_FILE) -> pd.DataFrame:
    """
    Run the VANRIJN model using parameters from `input_excel`.
    Returns a DataFrame with columns: time, Qb, Z, B, B1, H, Hm, Hw, dt.
    """
    # locals used by original script will populate here
    # BEGIN original code (trimmed of plotting/backends)
    import matplotlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from math import atan
    from numpy import pi, tan, sqrt, sin, cos, log

    # 清除警告（类似 MATLAB 的 warning off）
    import warnings

    warnings.filterwarnings('ignore')

    # 读取 Excel 文件
    excel_file_path = str(input_excel)
    df_sheet1 = pd.read_excel(excel_file_path, sheet_name=0, header=None)
    df_sheet2 = pd.read_excel(excel_file_path, sheet_name=1, header=None)

    # 提取参数
    Cd = df_sheet1.iloc[0, 1]  # 流量系数
    Hs = df_sheet1.iloc[1, 1]  # 初始总高
    QIN = df_sheet2.values[:, 0]  # 入流量（一维数组）
    Hw0 = df_sheet1.iloc[3, 1]  # 湖水水位
    Z0 = df_sheet1.iloc[4, 1]  # 溃口底部标高
    Por0 = df_sheet1.iloc[5, 1:5].values.flatten()  # 孔隙率
    B0 = df_sheet1.iloc[7, 1]  # 溃口底部宽度
    d30 = df_sheet1.iloc[8, 1]  # d30
    D50 = df_sheet1.iloc[9, 1:5].values.flatten()  # D50
    MI = D50[0] ** (1 / 6) / 12  # 曼宁粗糙系数
    d90 = df_sheet1.iloc[10, 1]  # d90
    bt_0 = df_sheet1.iloc[11, 1]  # 初始坝体背水坡斜坡倾角
    bt_end = df_sheet1.iloc[12, 1]  # 坝体背水坡临界倾角
    bt_up = df_sheet1.iloc[13, 1]  # 坝体迎水坡脚
    Z_end = df_sheet1.iloc[14, 1]  # 溃坝临界高度
    Z_1 = df_sheet1.iloc[15, 1]  # 坝体底部相对基岩高程
    Db = df_sheet1.iloc[16, 1]  # 基岩高度
    L_1 = df_sheet1.iloc[17, 1]  # 坝体底部宽度
    step = df_sheet1.iloc[28, 1]  # 时间步长
    n = int(2 * df_sheet1.iloc[19, 1] / step)  # 循环次数
    dt = df_sheet1.iloc[20, 1]  # 时间增量
    fica = df_sheet1.iloc[21, 1]  # 方程中变量平方前系数
    ficb = df_sheet1.iloc[22, 1]  # 方程中变量前系数
    ficc = df_sheet1.iloc[23, 1]  # 方程中常数
    m = df_sheet1.iloc[30, 1]  # 转换系数
    stage1 = df_sheet1.iloc[35, 1]  # 堰塞坝材料分层一阶段
    stage2 = df_sheet1.iloc[36, 1]  # 堰塞坝材料分层二阶段
    stage3 = df_sheet1.iloc[37, 1]  # 堰塞坝材料分层三阶段
    state = int(df_sheet1.iloc[49, 1])  # 分层
    pw = df_sheet1.iloc[31, 1]  # 水容重
    Ps = df_sheet1.iloc[32, 1:5].values.flatten()  # 颗粒容重
    FAI = df_sheet1.iloc[33, 1:5].values.flatten()  # 内摩擦角
    C = df_sheet1.iloc[25, 1]  # 粘聚力
    v = float(1.1400e-06)  # 运动粘度系数
    g = float(9.81)  # 重力加速度

    # 初始化（所有数组改为 1D，形状为 (n,) 或 (n+1,)）
    As0 = fica * ((Hw0 + Db) ** 2) - ficb * (Hw0 + Db) + ficc
    Hm0 = Hs - Z0
    a = np.ones(n + 1) * pi / 2
    Hm = np.ones(n) * Hm0
    t = np.zeros(n)
    Qb = np.zeros(n)
    Z = np.ones(n) * Z0
    Hw = np.ones(n) * Hw0
    As = np.ones(n) * As0
    B = np.ones(n) * B0
    B1 = np.ones(n) * B0
    number = np.zeros(n)
    seita = np.ones(n) * pi / 2
    seita_s = np.zeros(n)
    y = np.zeros(n)
    u = np.zeros(n)
    A = np.zeros(n)
    N = np.zeros(n, dtype=int)
    Qin = np.zeros(n)
    h = np.zeros(n)
    Fd = np.zeros(n)
    Fr = np.zeros(n)
    H = np.zeros(n + 1)
    seita_s = np.zeros(n)
    seita = np.ones(n + 1) * pi / 2
    Lx = np.zeros(n + 1)
    L = np.zeros(n + 1)
    bt = np.zeros(n + 1)
    E = np.zeros(n + 1)
    shields = np.zeros(n + 1)
    shields_c = np.zeros(n + 1)
    taob = np.zeros(n + 1)
    taoc = np.zeros(n + 1)
    c_ = np.zeros(n + 1)

    # while 循环
    i = 2
    Shields = 0.0
    Shields_c = 1.0
    while Shields < Shields_c and i <= n:
        if state == 888:
            if Hm[i - 1] < stage1:
                d50 = D50[0]
                Por = Por0[0]
                ps = Ps[0]
                fai = FAI[0]
            elif stage1 <= Hm[i - 1] < stage2:
                d50 = D50[1]
                Por = Por0[1]
                ps = Ps[1]
                fai = FAI[1]
            elif stage2 <= Hm[i - 1] < stage3:
                d50 = D50[2]
                Por = Por0[2]
                ps = Ps[2]
                fai = FAI[2]
            else:
                d50 = D50[3]
                Por = Por0[3]
                ps = Ps[3]
                fai = FAI[3]
        else:
            d50 = D50[1]
            Por = Por0[0]
            ps = Ps[0]
            fai = FAI[1]

        t[i - 1] = dt * step * (i - 1)
        y[i - 2] = m * (Hw[i - 2] - Z0)
        Qb[i - 1] = Cd * B0 * (Hw[i - 2] - Z0) ** 1.5
        A[i - 2] = fica * ((Hw[i - 2] + Db) ** 2) - ficb * (Hw[i - 2] + Db) + ficc
        dl_H = (t[i - 1] - t[i - 2]) / A[i - 2] * (QIN[0] - Qb[i - 1])
        Hw[i - 1] = Hw[i - 2] + dl_H
        u[i - 2] = Cd * m ** (-1) * (Hw[i - 1] - Z0) ** 0.5
        # c_ = 5.75 * sqrt(g) * log(12 * y[i - 2] / (3 * d90))
        taob_ = g * pw * MI ** 2 * u[i - 2] ** 2 / (y[i - 2] ** (1 / 3))
        Shields = taob_ / ((ps - pw) * g * d50)
        taoc_ = 2 / 3 * g * d50 * (ps - pw) * tan(fai)
        Shields_c = taoc_ / ((2650 - pw) * g * d50)
        i += 1
        nn = i

    # 主循环
    for i in range(nn, n + 1):
        t[i - 1] = dt * (i - 1) * step
        if state == 888:
            if Hm[i - 2] < stage1:
                d50 = D50[0]
                Por = Por0[0]
                ps = Ps[0]
                fai = FAI[0]
            elif stage1 <= Hm[i - 2] < stage2:
                d50 = D50[1]
                Por = Por0[1]
                ps = Ps[1]
                fai = FAI[1]
            elif stage2 <= Hm[i - 2] < stage3:
                d50 = D50[2]
                Por = Por0[2]
                ps = Ps[2]
                fai = FAI[2]
            else:
                d50 = D50[3]
                Por = Por0[3]
                ps = Ps[3]
                fai = FAI[3]
        else:
            d50 = D50[1]
            Por = Por0[0]
            ps = Ps[0]
            fai = FAI[0]

        N[i - 2] = int(round((t[i - 1] + dt) / dt))
        Qin[i - 1] = QIN[N[i - 2] - 1] + (QIN[N[i - 2]] - QIN[N[i - 2] - 1]) * \
                     (t[i - 1] / dt - step * i) / (step * (i + 1) - t[i - 1] / dt)
        h[i - 2] = m * (Hw[i - 2] - Z[i - 2])

        if Hw[i - 2] <= Z[i - 2]:
            Qb[i - 2] = 0
        elif seita[i - 1] == pi / 2:
            Qb[i - 2] = Cd * B[i - 2] * (Hw[i - 2] - Z[i - 2]) ** 1.5
            if Qb[i - 2] < 0:
                Qb[i - 2] = 0
        else:
            Qb[i - 2] = 1.7 * B[i - 2] * (Hw[i - 2] - Z[i - 2]) ** 1.5 + \
                        1.3 * tan(seita[i - 1]) * (Hw[i - 2] - Z[i - 2]) ** 2.5
            if Qb[i - 2] < 0:
                Qb[i - 2] = 0

        dl_Hw = (Qin[i - 2] - Qb[i - 2]) / As[i - 2] * (t[i - 1] - t[i - 2])
        Hw[i - 1] = Hw[i - 2] + dl_Hw
        if Hw[i - 1] <= 0:
            Hw[i - 1] = 0
        As[i - 1] = fica * ((Hw[i - 1] + Db) ** 2) - ficb * (Hw[i - 1] + Db) + ficc
        bt[i - 2] = bt_0 - ((Z0 - Z[i - 2]) / (Z0 - Z_end)) * (bt_0 - bt_end)
        if bt[i - 2] <= bt_end:
            bt[i - 2] = bt_end
        u[i - 2] = Cd * m ** (-1) * (Hw[i - 2] - Z[i - 2]) ** 0.5
        c_[i - 2] = 5.75 * sqrt(g) * log(12 * h[i - 2] / (3 * d90))
        taob[i - 2] = pw * g * (u[i - 2] / c_[i - 2]) ** 2
        shields[i - 2] = taob[i - 2] / ((ps - pw) * g * d50)
        taoc[i - 2] = 2 / 3 * g * d50 * (ps - pw) * tan(fai)
        shields_c[i - 2] = taoc[i - 2] / ((2650 - pw) * g * d50)
        bt[i - 1] = bt_0 - ((Z0 - Z[i - 2]) / (Z0 - Z_end)) * (bt_0 - bt_end)
        if bt[i - 1] <= bt_end:
            bt[i - 1] = bt_end
        Lx[i - 2] = Z[i - 2] / sin(bt[i - 1])
        L[i - 2] = 320 + 2 * (Z0 - Z[i - 2]) / tan(bt_0)

        if shields[i - 2] > 1:
            fd = 1 / shields[i - 2]
        else:
            fd = 1

        D_ = d50 * ((ps / pw - 1) * g / v ** 2) ** (1 / 3)

        if shields[i - 2] > shields_c[i - 2]:
            E[i - 1] = 0.00033 * ps * ((ps / pw - 1) * g * d50) ** 0.5 * (D_) ** 0.3 * fd * \
                       ((shields[i - 2] - shields_c[i - 2]) / shields_c[i - 2]) ** 1.5
        else:
            E[i - 1] = 0

        if E[i - 1] > 0:
            dl_Z = E[i - 1] * (t[i - 1] - t[i - 2]) / (2650 * (1 - Por))
        else:
            dl_Z = 0

        Z[i - 1] = Z[i - 2] - dl_Z
        if Z[i - 1] < 0:
            Z[i - 1] = 0

        Fd[i - 1] = 0.5 * ps * g * (Hs - Z[i - 1]) ** 2 * (1 / tan(a[i - 1]) - 1 / tan(seita[i - 1])) * sin(a[i - 1])
        Fr[i - 1] = 0.5 * ps * g * (Hs - Z[i - 1]) ** 2 * (1 / tan(a[i - 1]) - 1 / tan(seita[i - 1])) * cos(a[i - 1]) * tan(
            fai) + \
                    (C * (Hs - Z[i - 1])) / sin(a[i - 1])

        H[i] = Hs - Z[i - 1]

        if Fd[i - 1] > Fr[i - 1]:
            B[i - 1] = B[i - 2] + 2 * dl_Z * (1 / sin(seita[i - 1]) - 1 / tan(seita[i - 1])) + \
                       ((Hs - Z[i - 1]) - (Hs - Z[i - 2])) / tan(seita[i - 1])
            B1[i - 1] = B1[i - 2] + 2 * dl_Z * 1 / sin(seita[i - 1]) + \
                        ((Hs - Z[i - 1]) - (Hs - Z[i - 2])) / tan(seita[i - 1])
            seita_s[i - 1] = atan((tan(fai) ** 2) / (tan(fai) + (4 * C / (ps * H[i]) - \
                                                                 sqrt((16 * C ** 2) / (ps * H[i]) ** 2 + 8 * C / (
                                                                             ps * H[i]) * tan(fai)) * \
                                                                 (1 + tan(fai) ** 2))))
            a[i] = atan(0.5 * (1 + tan(fai) / tan(seita[i - 1])) / (2 * C / (ps * H[i]) + 1 / tan(seita[i - 1])))
            seita[i] = (seita_s[i - 1] + a[i]) / 2
        else:
            B[i - 1] = B[i - 2] + 2 * dl_Z * (1 / sin(seita[i - 1]) - 1 / tan(seita[i - 1]))
            B1[i - 1] = B1[i - 2] + 2 * dl_Z * 1 / sin(seita[i - 1])
            seita_s[i - 1] = atan((tan(fai) ** 2) / (tan(fai) + (4 * C / (ps * H[i]) - \
                                                                 sqrt((16 * C ** 2) / (ps * H[i]) ** 2 + 8 * C / (
                                                                             ps * H[i]) * tan(fai)) * \
                                                                 (1 + tan(fai) ** 2))))
            a[i] = atan(0.5 * (1 + tan(fai) / tan(seita[i - 1])) / (2 * C / (ps * H[i]) + 1 / tan(seita[i - 1])))
            seita[i] = seita[i - 1]

        Hm[i - 1] = Hs - Z[i - 1]



    # Compose DataFrame from available arrays
    out = pd.DataFrame({
        "time": pd.Series(t).astype(float),
        "Qb": pd.Series(Qb).astype(float),
        "Z": pd.Series(Z).astype(float),
        "B": pd.Series(B[:-1] if len(B)>len(Qb) else B).astype(float) if 'B' in locals() else np.nan,
        "B1": pd.Series(B1[:-1] if 'B1' in locals() and len(B1)>len(Qb) else B1).astype(float) if 'B1' in locals() else np.nan,
        "H": pd.Series(H[:len(Qb)+1] if 'H' in locals() else np.nan).shift(-1) if 'H' in locals() else np.nan,
        "Hm": pd.Series(Hm).astype(float) if 'Hm' in locals() else np.nan,
        "Hw": pd.Series(Hw[:len(Qb)] if 'Hw' in locals() else np.nan) if 'Hw' in locals() else np.nan,
    })
    out["dt"] = float(dt) if "dt" in locals() else np.nan
    return out
