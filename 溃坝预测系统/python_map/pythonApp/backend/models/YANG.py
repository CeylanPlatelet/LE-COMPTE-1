import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi



import warnings
warnings.filterwarnings('ignore')

params_file = 'myFloodRoutingSys/dam/input/params3.xlsx'
params = pd.read_excel(params_file, sheet_name=0, header=None)
params2 = pd.read_excel(params_file, sheet_name=1, header=None)
step = params.iloc[28, 1]  # 时间步长
n = int(2 * params.iloc[19, 1] / step)  # 循环次数
Dt = params.iloc[1, 1]  # 初始总高m(坝体顶部高程)
QIN = params2.values[:, 0]  # 入流量
Hw0 = params.iloc[3, 1]  # 湖水水位(初始水位高程)
Z0 = params.iloc[4, 1]  # 溃口底部标高
B0 = params.iloc[7, 1]  # 溃口底部宽度(溃口宽度)
Db = params.iloc[16, 1]  # 基岩高度m(坝体底部高程)
fica = params.iloc[21, 1]  # 方程中变量平方前系数
ficb = params.iloc[22, 1]  # 方程中变量前系数
ficc = params.iloc[23, 1]  # 方程中常数
H0 = Dt - Z0  # 溃口深度
m = params.iloc[30, 1]  # 转换系数
dt = params.iloc[20, 1]
d50 = params.iloc[9, 1:5].values.flatten()
MI = d50[1] ** (1/6) / 12  # 曼宁粗糙系数
pw = params.iloc[31, 1]  # 水容重kg/m^3
ps = params.iloc[32, 1:5].values.flatten()  # 颗粒容重kg/m^3
Por0 = params.iloc[5, 1:5].values.flatten()  # 孔隙率
FAI = params.iloc[33, 1:5].values.flatten()  # 内摩擦角
stage1 = params.iloc[35, 1]  # 堰塞坝材料分层一阶段
stage2 = params.iloc[36, 1]  # 堰塞坝材料分层二阶段
stage3 = params.iloc[37, 1]  # 堰塞坝材料分层三阶段
c = params.iloc[25, 1]  # 粘聚力
Cd = params.iloc[0, 1]
UC = params.iloc[48, 1]  # 不确定性系数
state = params.iloc[49, 1]  # 分层
bt_0 = params.iloc[11, 1]  # 背水坡坡角
bt_end = params.iloc[12, 1]  # 背水坡坡角临界角度
bt_up = params.iloc[13, 1]  # 迎水坡坡角
Z_end = params.iloc[14, 1]  # 临界高度
v = 1.1400e-06  # 运动粘度系数
L_1 = params.iloc[17, 1]  # 坝体底部宽度
d90 = params.iloc[10, 1]  #
g = 9.81

# 初始化数组
Hw = np.ones(n+1) * Hw0
Z = np.ones(n+1) * Z0
B = np.ones(n+1) * B0
B1 = np.ones(n+1) * B0
H = np.ones(n+1) * H0
Qb = np.zeros(n)
t = np.zeros(n)
zd = np.zeros(n)
seita = np.ones(n+1) * (pi / 2)
a = np.ones(n+1) * ((pi / 4 + FAI[1]) / 2)

# while 循环
i = 1
Shields = 0.0
Shields_c = 1.0
while Shields < Shields_c and i <= n:
    if state == 888:
        if H[i-1] < stage1:
            D50 = d50[0]
            Por = Por0[0]
            Ps = ps[0]
            fai = FAI[0]
        elif stage1 <= H[i-1] < stage2:
            D50 = d50[1]
            Por = Por0[1]
            Ps = ps[1]
            fai = FAI[1]
        elif stage2 <= H[i-1] < stage3:
            D50 = d50[2]
            Por = Por0[2]
            Ps = ps[2]
            fai = FAI[2]
        else:
            D50 = d50[3]
            Por = Por0[3]
            Ps = ps[3]
            fai = FAI[3]
    else:
        D50 = d50[1]
        Por = Por0[0]
        Ps = ps[0]
        fai = FAI[0]

    t[i] = dt * step * i
    y = np.zeros(n)
    y[i-1] = m * (Hw[i-1] - Z0)
    Qb[i] = Cd * B0 * (Hw[i-1] - Z0) ** 1.5
    A = np.zeros(n)
    A[i-1] = fica * ((Hw[i-1] + Db) ** 2) - ficb * (Hw[i-1] + Db) + ficc
    dl_H = (t[i] - t[i-1]) / A[i-1] * (QIN[0] - Qb[i])
    Hw[i] = Hw[i-1] + dl_H
    u = np.zeros(n)
    u[i-1] = Cd * m ** (-1) * (Hw[i] - Z0) ** 0.5
    c_ = 5.75 * np.sqrt(9.81) * np.log(12 * y[i-1] / (3 * d90))
    # taob = g * pw * MI ** 2 * u[i-1] ** 2 / (y[i-1] ** (1/3))
    taob = pw * 9.81 * (u[i-1] / c_) ** 2
    Shields = taob / ((Ps - pw) * 9.81 * D50)
    taoc = 2 / 3 * g * D50 * (Ps - pw) * np.tan(fai)
    Shields_c = taoc / ((2650 - pw) * g * D50)
    nn = i
    i += 1
N = np.zeros(n)
Qin = np.zeros(n)
U = np.zeros(n)
h = np.zeros(n)
wm = np.zeros(n)
S = np.zeros(n)
U_ = np.zeros(n)
Critical_C = np.zeros(n)
Vcr = np.zeros(n)
ct = np.zeros(n)
bt = np.zeros(n)
L = np.zeros(n)
sr = np.zeros(n)
K1 = np.zeros(n + 1)
K2 = np.zeros(n + 1)
FOS = np.zeros(n + 1)
E1 = np.zeros(n + 1)
seita_s = np.zeros(n)
f = np.zeros(n)
# 主循环
for i in range(nn, n):
    t[i] = i * step * dt
    if state == 888:
        if H[i] < stage1:
            D50 = d50[0]
            e = Por0[0]
            Ps = ps[0]
            fai = FAI[0]
        elif stage1 <= H[i] < stage2:
            D50 = d50[1]
            e = Por0[1]
            Ps = ps[1]
            fai = FAI[1]
        elif stage2 <= H[i] < stage3:
            D50 = d50[2]
            e = Por0[2]
            Ps = ps[2]
            fai = FAI[2]
        else:
            D50 = d50[3]
            e = Por0[3]
            Ps = ps[3]
            fai = FAI[3]
    else:
        D50 = d50[1]
        e = Por0[0]
        Ps = ps[0]
        fai = FAI[0]

    N[i] = round((t[i] + dt) / dt)
    Qin[i] = QIN[int(N[i]) - 1] + (QIN[int(N[i])] - QIN[int(N[i]) - 1]) * \
                (t[i] / dt - step * (i + 1)) / (step * (i + 2) - t[i] / dt)

    c1 = 1.7
    c2 = 1.3
    Qb[i] = c1 * B[i] * (Hw[i] - Z[i]) ** 1.5 + c2 * np.tan(np.pi / 2 - seita[i]) * (Hw[i] - Z[i]) ** 2.5

    if Qb[i-1] < 0:
        Qb[i-1] = 0

    h[i] = (Hw[i] - Z[i]) * m

    DD50 = D50
    gg = g
    vv = v

    if D50 <= 1e-4:
        wm[i] = (2650 / pw - 1) * gg * DD50 ** 2 / (18 * vv)
    elif 1e-4 < D50 <= 1e-3:
        D_ = ((2650 / pw - 1) * gg / vv ** 2) ** (1/3) * DD50
        wm[i] = 10 * vv / DD50 * (np.sqrt(1 + 0.01 * D_ ** 3) - 1)
    else:
        wm[i] = 1.1 * np.sqrt((2650 / pw - 1) * gg * DD50)

    U[i] = Cd * np.sqrt(Hw[i] - Z[i])

    S[i] = (MI * U[i] / (h[i] ** (2/3))) ** 2

    U_[i] = np.sqrt(g * h[i] * S[i])

    Critical_C[i] = U_[i] * D50 / v

    if 1.2 < Critical_C[i] < 70:
        Vcr[i] = (2.5 * (np.log(U_[i] * D50 / v) - 0.06) ** (-1) + 0.66) * wm[i]
    else:
        Vcr[i] = 2.05 * wm[i]

    ct[i] = np.exp(5.435 - 0.286 * np.log(wm[i] * D50 / v) - 0.457 * np.log(U_[i] / wm[i]) +
                   (1.799 - 0.409 * np.log(wm[i] * D50 / v) - 0.314 * np.log(U_[i] / wm[i])) *
                   np.log(U[i] * S[i] / wm[i] - Vcr[i] * S[i] / wm[i]))

    bt[i] = bt_0 - ((Z0 - Z[i]) / (Z0 - Z_end)) * (bt_0 - bt_end)
    if bt[i] <= bt_end:
        bt[i] = bt_end

    L[i] = 400 + 2 * (Z0 - Z[i]) / np.tan(bt_0)

    sr[i] = ct[i] * Qb[i] * 1 / 1e6 * pw / (B[i] * L[i])

    if h[i] > 0 and sr[i] > 0:
        E1[i] = sr[i] * (t[i] - t[i-1]) / (Ps * (1 - e))
    else:
        E1[i] = 0

    if H[i] >= Dt:
        H[i + 1] = H[i]
    else:
        H[i + 1] = H[i] + E1[i]

    A[i] = fica * ((Hw[i] + Db) ** 2) - ficb * (Hw[i] + Db) + ficc
    Hw[i + 1] = Hw[i] + (Qin[i] - Qb[i]) / A[i] * (t[i] - t[i-1])
    Z[i + 1] = Z[i] - (H[i + 1] - H[i])

    if Z[i + 1] < 0:
        Z[i + 1] = 0

    K1[i + 1] = c * H[i] / np.sin(a[i]) + 0.5 * Ps * H[i] ** 2 * (1 / np.tan(a[i]) - 1 / np.tan(seita[i])) * np.cos(a[i]) * np.tan(fai)
    K2[i + 1] = Ps * 0.5 * H[i] ** 2 * (1 / np.tan(a[i]) - 1 / np.tan(seita[i])) * np.sin(a[i])
    FOS[i + 1] = K1[i + 1] / K2[i + 1]

    f[i] = 1 / (1 + np.exp(UC * (FOS[i + 1] - 1)))

    if 0 <= f[i] <= 1:
        B[i + 1] = B[i] + 2 * (H[i + 1] - H[i]) * (1 / np.sin(seita[i]) - 1 / np.tan(seita[i])) + 1 * (H[i + 1] - H[i]) * (1 / np.tan(seita[i]))
        B1[i + 1] = B1[i] + 2 * (H[i + 1] - H[i]) * (1 / np.sin(seita[i])) + 1 * (H[i + 1] - H[i]) * (1 / np.tan(seita[i]))
        seita_s[i] = np.arctan((np.tan(fai) ** 2) / (np.tan(fai) + (4 * c / (Ps * H[i + 1]) - np.sqrt((16 * c ** 2) / (Ps * H[i + 1]) ** 2 + 8 * c / (Ps * H[i + 1]) * np.tan(fai)) * (1 + np.tan(fai) ** 2))))
        a[i + 1] = np.arctan(0.5 * (1 + np.tan(fai) / np.tan(seita[i])) / (2 * c / (Ps * H[i + 1]) + 1 / np.tan(seita[i])))
        seita[i + 1] = (seita_s[i] + a[i + 1]) / 2
    else:
        B[i + 1] = B[i] + 2 * (H[i + 1] - H[i]) * (1 / np.sin(seita[i]) - 1 / np.tan(seita[i]))
        B1[i + 1] = B1[i] + 2 * (H[i + 1] - H[i]) * (1 / np.sin(seita[i]))
        seita_s[i] = np.arctan((np.tan(fai) ** 2) / (np.tan(fai) + (4 * c / (Ps * H[i + 1]) - np.sqrt((16 * c ** 2) / (Ps * H[i + 1]) ** 2 + 8 * c / (Ps * H[i + 1]) * np.tan(fai)) * (1 + np.tan(fai) ** 2))))
        a[i + 1] = np.arctan(0.5 * (1 + np.tan(fai) / np.tan(seita[i])) / (2 * c / (Ps * H[i + 1]) + 1 / np.tan(seita[i])))
        seita[i + 1] = seita[i]

# 绘图
if len(t) != len(Qb):
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