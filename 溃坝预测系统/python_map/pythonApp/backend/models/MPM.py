import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

params_file = 'myFloodRoutingSys/dam/input/params3.xlsx'
params = pd.read_excel(params_file, sheet_name=0, header=None)
params2 = pd.read_excel(params_file, sheet_name=1, header=None)
Cd = params.iloc[0, 1]  # 流量系数m^0.5/s
Hs = params.iloc[1, 1]  # 初始总高m(坝体顶部高程)
QIN = params2.values[:, 0]  # 入流量
Hw0 = params.iloc[3, 1]  # 湖水水位(初始水位高程)
Z0 = params.iloc[4, 1]  # 溃口底部标高
Por0 = params.iloc[5, 1:5].values.flatten()  # 孔隙率
B0 = params.iloc[7, 1]  # 溃口底部宽度(溃口宽度)
D50 = params.iloc[9, 1:5].values.flatten()
MI = D50[0] ** (1/6) / 12  # 曼宁粗糙系数
Z_end = params.iloc[14, 1]  # 溃坝临界高度
Z_1 = params.iloc[15, 1]  # 坝体底部相对基岩高程
Db = params.iloc[16, 1]  # 基岩高度m(坝体底部高程)
L_1 = params.iloc[17, 1]  # 坝体底部宽度
step = params.iloc[28, 1]  # 时间步长
n = int(1.5 * params.iloc[19, 1] / step)  # 循环次数
dt = params.iloc[20, 1]  # 时间步长
fica = params.iloc[21, 1]  # 方程中变量平方前系数
ficb = params.iloc[22, 1]  # 方程中变量前系数
ficc = params.iloc[23, 1]  # 方程中常数
v = 1.1400e-06  # 运动粘度系数
m = params.iloc[30, 1]  # 转换系数
pw = params.iloc[31, 1]  # 水容重kg/m^3
Ps = params.iloc[32, 1:5].values.flatten()  # 颗粒容重kg/m^3
bt_0 = params.iloc[11, 1]  # 背水坡坡角
bt_end = params.iloc[12, 1]  # 背水坡坡角临界角度
bt_up = params.iloc[13, 1]  # 迎水坡坡角
stage1 = params.iloc[35, 1]
stage2 = params.iloc[36, 1]
stage3 = params.iloc[37, 1]
state = params.iloc[49, 1]
d90 = params.iloc[10, 1] * 0.06 / 0.09
FAI = params.iloc[33, 1:5].values.flatten()  # 内摩擦角
g = 9.81

# 初始化
Hm0 = Hs - Z0
As0 = fica * ((Hw0 + Db) ** 2) - ficb * (Hw0 + Db) + ficc
Hm = np.ones(n) * Hm0
U = np.zeros(n)
t = np.zeros(n)
Qb = np.zeros(n)
Z = np.ones(n) * Z0
Hw = np.ones(n) * Hw0
As = np.ones(n) * As0
B = np.ones(n) * B0
B1 = np.ones(n) * B0
Qs = np.zeros(n)
bt = np.ones(n) * bt_0
Z1 = 0
rr = np.zeros(n)

# while 循环
i = 1
Shields = 0.0
Shields_c = 1.0
y = np.zeros(n)
P = np.zeros(n)
AA = np.zeros(n)
u = np.zeros(n)
while Shields < Shields_c and i < n:
    if state == 888:
        if Hm[i-1] < stage1:
            d50 = D50[0]
            Por = Por0[0]
            ps = Ps[0]
            fai = FAI[0]
        elif stage1 <= Hm[i-1] < stage2:
            d50 = D50[1]
            Por = Por0[1]
            ps = Ps[1]
            fai = FAI[1]
        elif stage2 <= Hm[i-1] < stage3:
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

    t[i] = dt * step * i
    y[i-1] = m * (Hw[i-1] - Z0)
    P[i-1] = B0 + 2 * y[i-1]
    Qb[i] = Cd * B0 * (Hw[i-1] - Z0) ** 1.5
    AA[i-1] = fica * ((Hw[i-1] + Db) ** 2) - ficb * (Hw[i-1] + Db) + ficc
    dl_H = (t[i] - t[i-1]) / AA[i-1] * (QIN[0] - Qb[i])
    Hw[i] = Hw[i-1] + dl_H
    u[i-1] = Cd * m ** (-1) * (Hw[i] - Z0) ** 0.5
    c_ = 5.75 * np.sqrt(9.81) * np.log(12 * y[i-1] / (3 * d90))
    taob_ = pw * 9.81 * (u[i-1] / c_) ** 2
    Shields = taob_ / ((ps - pw) * 9.81 * d50)
    taoc = 2 / 3 * g * d50 * (ps - pw) * np.tan(fai)
    Shields_c = taoc / ((2650 - pw) * g * d50)
    nn = i
    i += 1

# 主循环
N = np.zeros(n, dtype=int)
Qin = np.zeros(n)
rep = np.zeros(n)
h = np.zeros(n)
A = np.zeros(n)
HH = np.zeros(n)
ct_b = np.zeros(n)
tg = np.zeros(n)
taoc = np.zeros(n)
tg_c = np.zeros(n)
Lx = np.zeros(n)
L = np.zeros(n)
for i in range(nn, n):
    t[i] = dt * (i) * step
    if state == 888:
        if Hm[i-1] < stage1:
            d50 = D50[0]
            Por = Por0[0]
            ps = Ps[0]
            fai = FAI[0]
        elif stage1 <= Hm[i-1] < stage2:
            d50 = D50[1]
            Por = Por0[1]
            ps = Ps[1]
            fai = FAI[1]
        elif stage2 <= Hm[i-1] < stage3:
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

    rr[i-1] = Z[i-1] / Hs

    N[i-1] = round((t[i] + dt) / dt)
    Qin[i] = QIN[int(N[i-1]) - 1] + (QIN[int(N[i-1])] - QIN[int(N[i-1]) - 1]) * \
                (t[i] / dt - step * (i + 1)) / (step * (i + 2) - t[i] / dt)

    rep[i-1] = -2.82 * np.log(rr[i-1]) + 0.351  # coleman文章中k*
    h[i-1] = (Hw[i-1] - Z[i-1]) * m
    B[i-1] = 2 * rep[i-1] * np.sqrt((Hs - Z[i-1]) / 2) * np.sqrt(Z[i-1])
    if i > 1 and B[i-1] < B[i-2]:
        B[i-1] = B[i-2]
    B1[i-1] = 2 * rep[i-1] * np.sqrt(Hs - Z[i-1]) * np.sqrt(Z[i-1])
    if i > 1 and B1[i-1] < B1[i-2]:
        B1[i-1] = B1[i-2]
    A[i-1] = (4/3) * rep[i-1] * h[i-1] ** (3/2) * np.sqrt(Hs)
    HH[i-1] = Z[i-1] + 9800 * h[i-1] / 9800 + U[i-1] ** 2 / (2 * g)
    Qb[i-1] = Cd * A[i-1] * HH[i-1] ** (1/2)
    if Qb[i-1] < 0:
        Qb[i-1] = Qin[i]
    dl_Hw = (Qin[i-1] - Qb[i-1]) / As[i-1] * (t[i] - t[i-1])
    Hw[i] = Hw[i-1] + dl_Hw
    if Hw[i] < 0:
        Hw[i] = 0
    As[i] = fica * ((Hw[i] + Db) ** 2) - ficb * (Hw[i] + Db) + ficc
    P[i-1] = 2 * h[i-1] + B[i-1]
    U[i] = Qb[i-1] / A[i-1]
    ct_b[i-1] = pw * g * MI ** 2 * U[i] ** 2 / (h[i-1] ** (1/3))
    tg[i-1] = ct_b[i-1] / ((ps - pw) * g * d50)
    bt[i-1] = bt_0 - ((Z0 - Z[i-1]) / (Z0 - Z_end)) * (bt_0 - bt_end)
    if bt[i-1] <= bt_end:
        bt[i-1] = bt_end
    taoc[i-1] = 2 / 3 * g * d50 * (2650 - pw) * np.tan(fai)
    tg_c[i-1] = taoc[i-1] / ((2650 - pw) * g * d50)
    Lx[i-1] = Z[i-1] / np.sin(bt[i-1])
    L[i-1] = 400 + 2 * (Z0 - Z[i-1]) / np.tan(bt_0)
    if tg[i-1] > tg_c[i-1]:
        Qs[i] = 8 * B[i-1] * (tg[i-1] - tg_c[i-1]) ** (3/2) * np.sqrt(((ps / pw - 1) * g * (d50) ** 3))
    else:
        Qs[i] = 0
    if Qs[i] > 0:
        dl_Z = Qs[i] / (P[i-1] * L[i-1] * (1 - Por)) * (t[i] - t[i-1])
    else:
        dl_Z = 0
    Z[i] = Z[i-1] - dl_Z
    if Z[i] < 0:
        Z[i] = 0
    Hm[i] = Hs - Z[i]
    h[i] = (Hw[i] - Z[i]) * m
    rr[i] = Z[i] / Hs
    HH[i] = Z[i] + 9800 * h[i] / 9800 + U[i] ** 2 / (2 * g)
    rep[i] = -2.82 * np.log(rr[i]) + 0.351
    B[i] = 2 * rep[i] * np.sqrt((Hs - Z[i]) / 2) * np.sqrt(Z[i])
    if B[i] < B[i-1]:
        B[i] = B[i-1]
    B1[i] = 2 * rep[i] * np.sqrt(Hs - Z[i]) * np.sqrt(Z[i])
    if B1[i] < B1[i-1]:
        B1[i] = B1[i-1]
    A[i] = (4/3) * rep[i] * h[i] ** (3/2) * np.sqrt(Hs)
    Qb[i] = Cd * A[i] * HH[i] ** (1/2)
    if Qb[i] < 0:
        Qb[i] = Qin[i]

# 绘图
if len(t) != len(Qb):
    min_length = min(len(t), len(Qb))
    t = t[:min_length]
    Qb = Qb[:min_length]

plt.figure()
plt.plot(t / dt, Qb, linewidth=3.0)
plt.xlabel('time')
plt.ylabel('discharge')
plt.title('discharge line')
plt.show()