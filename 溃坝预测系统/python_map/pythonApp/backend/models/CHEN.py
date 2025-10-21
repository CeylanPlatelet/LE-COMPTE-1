import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import pi, log, sqrt, tan, sin, cos, atan, degrees, radians

params_file = 'myFloodRoutingSys/dam/input/params3.xlsx'
params = pd.read_excel(params_file, sheet_name=0, header=None)
params2 = pd.read_excel(params_file, sheet_name=1, header=None)
step = params.iloc[28, 1]  # B29
dt = params.iloc[20, 1]  # B21
n = int(2 * params.iloc[19, 1] / step)  # B20, 确保 n 是整数
Cd = params.iloc[0, 1]  # B1
Hw0 = params.iloc[3, 1]  # B4
Z0 = params.iloc[4, 1]  # B5
B0 = params.iloc[7, 1]  # B8
Db = params.iloc[16, 1]  # B17
fica = params.iloc[21, 1]  # B22
ficb = params.iloc[22, 1]  # B23
ficc = params.iloc[23, 1]  # B24
m = params.iloc[30, 1]  # B31
a_param = params.iloc[39, 1]  # B40
b_param = params.iloc[40, 1]  # B41
pw = params.iloc[31, 1]  # B32
Ps = params.iloc[32, 1:5].values.flatten()  # B33:E33
Fai = params.iloc[33, 1:5].values.flatten()  # B34:E34
c = params.iloc[25, 1]  # B26
Hs = params.iloc[1, 1]  # B2
D50 = params.iloc[9, 1:5].values.flatten()  # B10:E10
MI = D50[0] ** (1/6) / 12  # 曼宁粗糙系数
bt_0 = params.iloc[11, 1]  # B12
bt_end = params.iloc[12, 1]  # B13
bt_up = params.iloc[13, 1]  # B14
Z_end = params.iloc[14, 1]  # B15
L_1 = params.iloc[17, 1]  # B18
Por0 = params.iloc[5, 1:5].values.flatten()  # B6:E6
state = params.iloc[49, 1]  # B50
stage1 = params.iloc[35, 1]  # B36
stage2 = params.iloc[36, 1]  # B37
stage3 = params.iloc[37, 1]  # B38
d90 = params.iloc[10, 1]  # B11
Hm0 = Hs - Z0
QIN = params2.values[:, 0]  # 入流量

# 初始化
t = np.zeros(n)
Hm = np.ones(n) * Hm0
N = np.zeros(n)
Qin = np.zeros(n)
h = np.zeros(n)
Qb = np.zeros(n)
Hw = np.ones(n+1) * Hw0
Z = np.ones(n+1) * Z0
B = np.ones(n+1) * B0
bt = np.ones(n) * bt_0
beita = np.ones(n) * (3/4 * pi - Fai[0] / 2)
y = np.zeros(n)
A = np.zeros(n)
u = np.zeros(n)
tao = np.zeros(n)
E = np.zeros(n)
B1 = np.ones(n+1) * B0
Lx = np.zeros(n)
L = np.zeros(n)

# while 循环
i = 2
Shields = 0.0
Shields_c = 1.0

while Shields < Shields_c and i <= n:
    if state == 888:
        if Hm[i-1] < stage1:
            d50 = D50[0]
            Por = Por0[0]
            ps = Ps[0]
            fai = Fai[0]
        elif stage1 <= Hm[i-1] < stage2:
            d50 = D50[1]
            Por = Por0[1]
            ps = Ps[1]
            fai = Fai[1]
        elif stage2 <= Hm[i-1] < stage3:
            d50 = D50[2]
            Por = Por0[2]
            ps = Ps[2]
            fai = Fai[2]
        else:
            d50 = D50[3]
            Por = Por0[3]
            ps = Ps[3]
            fai = Fai[3]
    else:
        d50 = D50[1]
        Por = Por0[0]
        ps = Ps[0]
        fai = Fai[1]

    t[i-1] = dt * step * (i-1)
    y[i-1] = m * (Hw[i-1] - Z0)
    Qb[i-1] = Cd * B0 * (Hw[i-1] - Z0) ** 1.5
    A[i-1] = fica * ((Hw[i-1] + Db) ** 2) - ficb * (Hw[i-1] + Db) + ficc
    dl_H = ((t[i-1] - t[i-2]) / A[i-1] * (QIN[0] - Qb[i-1])).item()
    Hw[i] = Hw[i-1] + dl_H
    u[i-1] = Cd * m ** (-1) * (Hw[i] - Z0) ** 0.5
    c_ = 5.75 * sqrt(9.81) * log(12 * y[i-1] / (3 * d90))
    # taob_ = 9.81 * pw * MI ** 2 * u[i - 1] ** 2 / (y[i - 1] ** (1 / 3))
    taob_ = pw * 9.81 * (u[i-1] / c_) ** 2
    Shields = taob_ / ((ps - pw) * 9.81 * d50)
    taoc_ = 2 / 3 * 9.81 * d50 * (ps - pw) * tan(fai)
    Shields_c = taoc_ / ((2650 - pw) * 9.81 * d50)
    i += 1
    nn = i

# for 循环
for i in range(nn, n):
    t[i] = i * step * dt
    if state == 888:
        if Hm[i] < stage1:
            d50 = D50[0]
            Por = Por0[0]
            ps = Ps[0]
            fai = Fai[0]
        elif stage1 <= Hm[i] < stage2:
            d50 = D50[1]
            Por = Por0[1]
            ps = Ps[1]
            fai = Fai[1]
        elif stage2 <= Hm[i] < stage3:
            d50 = D50[2]
            Por = Por0[2]
            ps = Ps[2]
            fai = Fai[2]
        else:
            d50 = D50[3]
            Por = Por0[3]
            ps = Ps[3]
            fai = Fai[3]
    else:
        d50 = D50[1]
        Por = Por0[0]
        ps = Ps[0]
        fai = Fai[1]

    N[i] = round((t[i] + dt) / dt)
    Qin[i] = QIN[int(N[i]) - 1] + (QIN[int(N[i])] - QIN[int(N[i]) - 1]) * \
                (t[i] / dt - step * (i + 1)) / (step * (i + 2) - t[i] / dt)


    taoc = 2 / 3 * 9.81 * d50 * (ps - pw) * tan(fai)

    Am2 = 0.0332 * tan(fai) + 0.0086
    Bm2 = 0.0421 * tan(fai) + 0.0042
    m2 = Am2 + 1 / 6 * (ps / 100 - 16) * (Bm2 - Am2)

    Am1 = 0.0058 * c + 0.0177
    Bm1 = 0.0073 * c + 0.0716
    Cm1 = 0.0149 * c + 0.1003
    acfai = fai

    if 0 < acfai <= radians(17):
        m11 = 0 + (acfai - 0) * (Am1 - 0) / (radians(17) - 0)
    elif radians(17) < acfai <= radians(27):
        m11 = Am1 + (acfai - radians(17)) * (Bm1 - Am1) / (radians(27) - radians(17))
    elif radians(27) < acfai <= radians(37):
        m11 = Bm1 + (acfai - radians(27)) * (Cm1 - Bm1) / (radians(37) - radians(27))
    else:
        raise ValueError('内摩擦角不在定义的区间内')

    RAm1 = 0.0041 * c + 0.0019
    RBm1 = 0.0077 * c - 0.0496
    RCm1 = 0.0092 * c + 0.0379

    if 0 < acfai <= radians(17):
        m12 = 0 + (acfai - 0) * (RAm1 - 0) / (radians(17) - 0)
    elif radians(17) < acfai <= radians(27):
        m12 = RAm1 + (acfai - radians(17)) * (RBm1 - RAm1) / (radians(27) - radians(17))
    elif radians(27) < acfai <= radians(37):
        m12 = RBm1 + (acfai - radians(27)) * (RCm1 - RBm1) / (radians(37) - radians(27))
    else:
        raise ValueError('内摩擦角不在定义的区间内')

    m1 = m11 + 1 / 6 * (ps / 100 - 16) * (m12 - m11)
    beita_end = (3 / 4 * pi - 0.5 * fai) + (Z0 - Z_end) / (m2 * (Z0 - Z_end) + m1)

    if Hw[i] <= Z[i]:
        Qb[i] = 0
    else:
        Qb[i] = Cd * B[i] * (Hw[i] - Z[i]) ** 1.5
        if Qb[i] < 0:
            Qb[i] = 0

    h[i] = m * (Hw[i] - Z[i])
    u[i] = Cd * m ** (-1) * (Hw[i] - Z[i]) ** 0.5
    tao[i] = 9.81 * pw * MI ** 2 * u[i] ** 2 / (h[i] ** (1 / 3))
    er = 100 * (tao[i] - taoc)
    if (tao[i] - taoc) < 0:
        er = 0

    E[i] = er / (a_param + b_param * er) * 10 ** (-3)
    A[i] = fica * ((Hw[i] + Db) ** 2) - ficb * (Hw[i] + Db) + ficc

    Hw[i + 1] = Hw[i] + ((Qin[i] - Qb[i]) / A[i] * (t[i] - t[i - 1])).item()
    bt[i] = bt_0 - ((Z0 - Z[i]) / (Z0 - Z_end)) * (bt_0 - bt_end)
    if bt[i] <= bt_end:
        bt[i] = bt_end
    Lx[i] = Z[i] / sin(bt[i])
    L[i] = 2 * (110 * tan(bt_0) + (Z0 - Z[i]) / tan(bt_0))

    if er > 0:
        Z[i + 1] = Z[i] - E[i] * (t[i] - t[i - 1]) / 1000
    else:
        Z[i + 1] = Z[i]

    if Z[i + 1] < 0:
        Z[i + 1] = 0

    beita[i] = (3 / 4 * pi - fai / 2) + (Hs - Z[i + 1]) / (m2 + m1 * (Hs - Z[i + 1]))
    if beita[i] > beita_end:
        beita[i] = beita_end

    B1[i + 1] = B0 + 2 * (Hs - Z[i + 1]) + 2 * (Hs - Z[i + 1]) * abs(tan(beita[i]))
    B[i + 1] = B0 + 2 * (Hs - Z[i + 1])
    if B[i + 1] < B[i]:
        B[i + 1] = B[i]

    Hm[i] = Hs - Z[i]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(t / dt, Qb, linewidth=3.0)
plt.xlabel('time')
plt.ylabel('discharge')
plt.title('discharge line')
plt.grid(True)
plt.show()