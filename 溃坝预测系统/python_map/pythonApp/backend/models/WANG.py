import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import pi, log, sqrt, tan, sin, cos

# 读取 Excel 数据
params_file = 'myFloodRoutingSys/dam/input/params3.xlsx'

params = pd.read_excel(params_file, sheet_name=0, header=None)
params2 = pd.read_excel(params_file, sheet_name=1, header=None)
step = params.iloc[28, 1]
n = int(2 * params.iloc[19, 1] / step)
Cd = params.iloc[0, 1]
Hw0 = params.iloc[3, 1]
Z0 = params.iloc[4, 1]
B0 = params.iloc[7, 1]
Db = params.iloc[16, 1]
fica = params.iloc[21, 1]
ficb = params.iloc[22, 1]
ficc = params.iloc[23, 1]
H0 = params.iloc[24, 1]
m = params.iloc[30, 1]
dt = params.iloc[20, 1]
d50 = params.iloc[9, 1:5].values.flatten()
MI = d50[1] ** (1 / 6) / 12  # 曼宁粗糙系数
pw = params.iloc[31, 1]
ps = params.iloc[32, 1:5].values.flatten()
Por = params.iloc[5, 1:5].values.flatten()
FAI = params.iloc[33, 1:5].values.flatten()
stage1 = params.iloc[35, 1]
stage2 = params.iloc[36, 1]
stage3 = params.iloc[37, 1]
Z_end = params.iloc[14, 1]
state = params.iloc[49, 1]
d90 = params.iloc[10, 1]
bt_0 = params.iloc[11, 1]
QIN = params2.values[:, 0]  # 入流量

# 初始化数组
Hw = np.ones(n) * Hw0
B = np.ones(n+1) * B0
B1 = np.ones(n+1) * B0
Z = np.ones(n+1) * Z0
Qb = np.zeros(n)
H = np.ones(n+1) * H0
t = np.zeros(n)
g = 9.81
y = np.zeros(n)
P = np.zeros(n)
As0 = fica * ((Hw0 + Db) ** 2) - ficb * (Hw0 + Db) + ficc
As = np.ones(n) * As0
u = np.zeros(n)
taob = np.zeros(n)
h = np.zeros(n)
sr = np.zeros(n)
E = np.zeros(n)
N = np.zeros(n)
Qin = np.zeros(n)

i = 2
Shields = 0.0
Shields_c = 1.0

while Shields < Shields_c:
    if state == 888:
        if H[i - 1] < stage1:
            D50 = d50[0]
            Por0 = Por[0]
            Ps = ps[0]
            fai = FAI[0]
        elif stage1 <= H[i - 1] < stage2:
            D50 = d50[1]
            Por0 = Por[1]
            Ps = ps[1]
            fai = FAI[1]
        elif stage2 <= H[i - 1] < stage3:
            D50 = d50[2]
            Por0 = Por[2]
            Ps = ps[2]
            fai = FAI[2]
        else:
            D50 = d50[3]
            Por0 = Por[3]
            Ps = ps[3]
            fai = FAI[3]
    else:
        D50 = d50[1]
        Por0 = Por[0]
        Ps = ps[0]
        fai = FAI[0]

    t[i - 1] = dt * step * (i - 1)
    y[i - 1] = m * (Hw[i - 1] - Z0)
    P[i - 1] = B0 + 2 * y[i - 1]
    Qb[i - 1] = Cd * B0 * (Hw[i - 1] - Z0) ** 1.5
    As[i - 1] = fica * ((Hw[i - 1] + Db) ** 2) - ficb * (Hw[i - 1] + Db) + ficc
    dl_H = (t[i - 1] - t[i - 2]) / As[i - 1] * (QIN[0] - Qb[i - 1])
    Hw[i] = Hw[i - 1] + dl_H
    u[i - 1] = Cd * m ** (-1) * (Hw[i] - Z0) ** 0.5
    c_ = 5.75 * sqrt(g) * log(12 * y[i - 1] / (3 * d90))
    #taob_ = g * pw * MI ** 2 * u[i - 1] ** 2 / (y[i - 1] ** (1 / 3))
    taob_ = pw * g * (u[i - 1] / c_) ** 2
    Shields = taob_ / ((Ps - pw) * g * D50)
    taoc_ = 2 / 3 * g * D50 * (Ps - pw) * tan(fai)
    Shields_c = taoc_ / ((2650 - pw) * g * D50)
    i += 1
    nn = i

# for 循环
for i in range(nn, n + 1):
    t[i - 1] = dt * i * step
    N[i - 1] = round((t[i - 1] + dt) / dt)
    Qin[i - 1] = QIN[int(N[i - 1]) - 1] + (QIN[int(N[i - 1])] - QIN[int(N[i - 1]) - 1]) * \
                (t[i - 1] / dt - step * i) / (step * (i + 1) - t[i - 1] / dt)

    if state == 888:
        if H[i - 1] < stage1:
            D50 = d50[0]
            e = Por[0]
            C = ps[0]
            fai = FAI[0]
        elif stage1 <= H[i - 1] < stage2:
            D50 = d50[1]
            e = Por[1]
            C = ps[1]
            fai = FAI[1]
        elif stage2 <= H[i - 1] < stage3:
            D50 = d50[2]
            e = Por[2]
            C = ps[2]
            fai = FAI[2]
        else:
            D50 = d50[3]
            e = Por[3]
            C = ps[3]
            fai = FAI[3]
    else:
        D50 = d50[1]
        e = Por[0]
        C = ps[0]
        fai = FAI[1]

    Qb[i - 1] = 1.7 * B[i - 1] * (Hw[i - 1] - Z[i - 1]) ** 1.5 + 1.3 * tan(1 / 1.3) * (
                Hw[i - 1] - Z[i - 1]) ** 2.5
    if Qb[i - 1] < 0:
        Qb[i - 1] = 0

    h[i - 1] = m * (Hw[i - 1] - Z[i - 1])
    u[i - 1] = Cd * m ** (-1) * sqrt(Hw[i - 1] - Z[i - 1])
    sr[i - 1] = 0.218 * (pw * g / (C * g - pw * g)) * MI * u[i - 1] / (D50 ** 0.25 * h[i - 1] ** 0.67) * \
                (pw * g * MI ** 2 * u[i - 1] ** 3 / h[i - 1] ** 0.33 - 0.1 * pw * (
                            (C * g - pw * g) / (pw * g) * g * D50) ** 1.5)
    E[i - 1] = sr[i - 1] / (C * 1 * e * B[i - 1]) * (t[i - 1] - t[i - 2])

    if E[i - 1] > 0:
        H[i] = H[i - 1] + E[i - 1]
        Z[i] = Z[i - 1] - E[i - 1]
        B[i] = B[i - 1] + 2 * E[i - 1]
        B1[i] = B1[i - 1] + 2 * E[i - 1]
    elif i < n:
        H[i] = H[i - 1]
        Z[i] = Z[i - 1]
        B[i] = B[i - 1]
        B1[i] = B1[i - 1] + 2 * E[i - 1]

    if i < n and Z[i] < 0:
        Z[i] = 0

    As[i - 1] = fica * ((Hw[i - 1] + Db) ** 2) - ficb * (Hw[i - 1] + Db) + ficc
    if i < n and As[i - 1] != 0:
        Hw[i] = Hw[i - 1] + ((Qin[i - 1] - Qb[i - 1]) / As[i - 1] * (t[i - 1] - t[i - 2])).item()
    elif i < n:
        Hw[i] = Hw[i - 1]
        print(f"警告")

# 绘图
min_length = min(len(t), len(Qb))
t = t[:min_length]
Qb = Qb[:min_length]

plt.figure(figsize=(10, 6))
plt.plot(t / dt, Qb, linewidth=3.0)
plt.xlabel('time')
plt.ylabel('discharge')
plt.title('discharge line')
plt.grid(True)
plt.show()