import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


file_path = 'myFloodRoutingSys/dam/input/params3.xlsx'

df_sheet1 = pd.read_excel(file_path, sheet_name=0, header=None)
df_sheet2 = pd.read_excel(file_path, sheet_name=1, header=None)

step = df_sheet1.iloc[28, 1]        # 时间步长
n = 2 * df_sheet1.iloc[19, 1] / step  # 循环次数
n = int(n)
Dt = 3.28 * df_sheet1.iloc[1, 1]    # 初始总高m(坝体顶部高程)
QIN = 35.315 * df_sheet2.values     # 入流量m3/s
Z0 = 3.28 * df_sheet1.iloc[4, 1]           # 溃口底部标高
Db = 3.28 * df_sheet1.iloc[16, 1]   # 基岩高度m(坝体底部高程)

# 方程系数
fica = df_sheet1.iloc[21, 2]
ficb = df_sheet1.iloc[22, 2]
ficc = df_sheet1.iloc[23, 2]
fica0 = df_sheet1.iloc[21, 1]
ficb0 = df_sheet1.iloc[22, 1]
ficc0 = df_sheet1.iloc[23, 1]

d30 = 1000 * df_sheet1.iloc[8, 1]

D50 = 1000 * df_sheet1.iloc[9, 1:5].values
d90 = 1000 * df_sheet1.iloc[10, 1]

Por0 = df_sheet1.iloc[5, 1:5].values    # 孔隙率
FAI0 = df_sheet1.iloc[33, 1:5].values   # 内摩擦角

Hw0 = 3.28 * df_sheet1.iloc[3, 1]       # 湖水水位(初始水位高程)
C = 20.885 * df_sheet1.iloc[25, 1]      # 黏聚力
Z_end = 3.28 * df_sheet1.iloc[14, 1]    # 溃坝临界高度
Z_1 = 3.28 * df_sheet1.iloc[15, 1]      # 坝体底部相对基岩高程
L_1 = 3.28 * df_sheet1.iloc[17, 1]      # 坝体底部宽度

Br = df_sheet1.iloc[41, 1]              # 漫顶系数
bt_0 = df_sheet1.iloc[11, 1]            # 背水坡坡度
bt_end = df_sheet1.iloc[12, 1]          # 背水坡临界坡度
bt_up = df_sheet1.iloc[13, 1]           # 迎水坡坡度

ps = df_sheet1.iloc[32, 1:5].values / 16.0185
pw = df_sheet1.iloc[31, 1] / 16.0185
dt = df_sheet1.iloc[20, 1]
m = df_sheet1.iloc[30, 1]               # 转换系数

# 堰塞坝材料分层
stage1 = 3.28 * df_sheet1.iloc[35, 1]
stage2 = 3.28 * df_sheet1.iloc[36, 1]
stage3 = 3.28 * df_sheet1.iloc[37, 1]
state = df_sheet1.iloc[49, 1]           # 分层状态

Cd = df_sheet1.iloc[0, 1]               # 流量系数
B0 = df_sheet1.iloc[7, 1]               # 溃口底部宽度


# 初始化数组
n = int(n)
# 经验公式计算曼宁系数
Mn = (D50[0])**(1/6) / 12
# 将角度转换为弧度
ct_0 = np.deg2rad(90)  # 溃口侧壁和水平面夹角
Sa0 = (fica * ((Hw0 + Db)**2) - ficb * (Hw0 + Db) + ficc)
Hm0 = Dt - Z0
Hm = np.full(n, Hm0)
U = np.zeros(n)
k = np.zeros(n)
Hk = np.zeros(n)
Z = np.full(n, Z0)
Hw = np.full(n, Hw0)
RR = np.zeros(n)
at = np.zeros(n)
y = np.zeros(n)
ct = np.full(n, ct_0)
Qb = np.zeros(n)
Sa = np.full(n, Sa0)
g = 9.81
Z1 = 0
dlVV = 0
Shields = 0
Shields_c = 1
Hw = Hw / 3.28
t = np.zeros(n)
A = np.zeros(n)
u = np.zeros(n)

# 循环
i = 1
while Shields < Shields_c:
    if state == 888:
        if Hm[i - 1] < stage1:
            d50 = D50[0] / 1000
            Por = Por0[0]
            R = ps[0] * 16.0185
            FAI = FAI0[0]
        elif stage1 <= Hm[i - 1] < stage2:
            d50 = D50[1] / 1000
            Por = Por0[1]
            R = ps[1] * 16.0185
            FAI = FAI0[1]
        elif stage2 <= Hm[i - 1] < stage3:
            d50 = D50[2] / 1000
            Por = Por0[2]
            R = ps[2] * 16.0185
            FAI = FAI0[2]
        elif Hm[i - 1] >= stage3:
            d50 = D50[3] / 1000
            Por = Por0[3]
            R = ps[3] * 16.0185
            FAI = FAI0[3]
    else:
        d50 = D50[1] / 1000
        Por = Por0[0]
        R = ps[0] * 16.0185
        FAI = FAI0[0]

    t[i] = dt * step * (i - 1)

    y[i - 1] = m * (Hw[i - 1] - Z0 / 3.28)
    # 计算溃口流量
    Qb[i] = Cd * B0 / 3.28 * (Hw[i - 1] - Z0 / 3.28) ** 1.5

    # 计算水库面积
    A[i - 1] = fica0 * ((Hw[i - 1] + Db / 3.28) ** 2) - ficb0 * (Hw[i - 1] + Db / 3.28) + ficc0

    # 计算水位变化
    dl_H = (t[i] - t[i - 1]) / A[i - 1] * (QIN[0, 0] / 35.315 - Qb[i])

    # 更新水位
    Hw[i] = Hw[i - 1] + dl_H
    # 计算流速
    u[i - 1] = Cd * m ** (-1) * (Hw[i] - Z0 / 3.28) ** 0.5

    c_ = 5.75 * np.sqrt(9.81) * np.log(12 * y[i - 1] / (3 * d90 / 1000))

    # 计算床面切应力
    taob = pw * 16.0185 * g * (u[i - 1] / c_) ** 2
    Shields = taob / ((R - pw * 16.0185) * g * d50)

    # 计算临界切应力
    taoc = 2 / 3 * g * d50 * (R - pw * 16.0185) * np.tan(FAI)  # FAI是角度，需要转弧度
    Shields_c = taoc / ((2650 - pw * 16.0185) * g * d50)

    # 更新循环计数器
    i += 1
# 循环结束
nn = i

# 转换单位
Hw = Hw * 3.28
Qb = Qb * 35.315
y = y * 3.28
Sa = A[nn-2] * (3.28**2) * np.ones(n)
Bo = B0 * np.ones(n)
S = np.full(n, 0.001)

N = np.zeros(n, dtype=int)
Qin = np.zeros(n)
B = np.zeros(n)
AA = np.zeros(n)
bt = np.zeros(n)
Lx = np.zeros(n)
L = np.zeros(n)
R_ = np.zeros(n)
P = np.zeros(n)
seita = np.zeros(n)
a_ = np.zeros(n)
Po = np.zeros(n)
Qs = np.zeros(n)
# Main
for i in range(nn, n):
    t[i] = dt * step * i

    if state == 888:
        if Hm[i - 1] < stage1:
            d50 = D50[0]
            Por = Por0[0]
            R = ps[0]
            FAI = FAI0[0]
        elif stage1 <= Hm[i - 1] < stage2:
            d50 = D50[1]
            Por = Por0[1]
            R = ps[1]
            FAI = FAI0[1]
        elif stage2 <= Hm[i - 1] < stage3:
            d50 = D50[2]
            Por = Por0[2]
            R = ps[0, 2]
            FAI = FAI0[2]
        elif Hm[i - 1] >= stage3:
            d50 = D50[3]
            Por = Por0[3]
            R = ps[3]
            FAI = FAI0[3]
    else:
        d50 = D50[0]
        Por = Por0[0]
        R = ps[0]
        FAI = FAI0[0]

    N[i - 1] = round((t[i] + dt) / dt)
    Qin[i] = QIN[int(N[i - 1]) - 1, 0] + (QIN[int(N[i - 1]), 0] - QIN[int(N[i - 1]) - 1, 0]) * \
                    (t[i] / dt - step * i) / (step * (i + 1) - t[i] / dt)

    ct_0 = np.deg2rad(90)  # 溃口侧壁和水平面夹角

    ct1 = (ct_0 + FAI) / 2
    ct2 = (ct1 + FAI) / 2
    ct3 = (ct2 + FAI) / 2

    # H1, H2, H3 计算（使用不带 g 的版本）
    den1 = R * (1 - np.cos(ct_0 - FAI))
    den2 = R * (1 - np.cos(ct1 - FAI))
    den3 = R * (1 - np.cos(ct2 - FAI))

    H1 = (4 * C * np.cos(FAI) * np.sin(ct_0)) / den1
    H2 = (4 * C * np.cos(FAI) * np.sin(ct1)) / den2
    H3 = (4 * C * np.cos(FAI) * np.sin(ct2)) / den3

    y[i - 1] = m * (Hw[i - 1] - Z[i - 1])
    Hk = (Dt - Z[i - 1]) - y[i - 1] / 3

    # 溃口边坡崩塌判断
    if Hk < H1 and RR[i - 1] == 0:
        ct[i - 1] = ct_0
    elif Hk >= H1 and H2 > Hk and RR[i - 1] != 2 and RR[i - 1] != 3:
        ct[i - 1] = ct1
        RR[i] = 1
    elif Hk >= H2 and H3 > Hk and RR[i - 1] != 3:
        ct[i - 1] = ct2
        RR[i] = 2
    else:  # Hk>=H3
        ct[i - 1] = ct3
        RR[i] = 3

    # 溃口侧壁和垂直底部线的夹角
    if ct[i - 1] == ct_0:
        at[i] = 0
    else:
        at[i] = np.pi / 2 - ct[i - 1]

    # 溃口宽度计算
    if RR[i] == 0 or RR[i] == 1:
        Bo[i - 1] = Br * y[i - 1]
        if i > nn:
            if Bo[i - 1] < Bo[i - 2]:
                Bo[i - 1] = Bo[i - 2]
        if Bo[i - 1] < B0:
            Bo[i - 1] = B0
    else:
        Bo[i - 1] = Bo[i - 2]
        if Bo[i - 1] < B0:
            Bo[i - 1] = B0

    # 宽计算
    B[i - 1] = Bo[i - 1] + 2 * y[i - 1] * np.tan(at[i])
    if i > nn:
        if B[i - 1] < B[i - 2]:
            B[i - 1] = B[i - 2]

    # 过流断面面积计算
    if Hk < H1:
        AA[i - 1] = Bo[i - 1] * y[i - 1]  # 矩形溃口
    else:
        AA[i - 1] = (Bo[i - 1] + B[i - 1]) * y[i - 1] / 2  # 梯形溃口

    # Qb (溃口流量) 计算
    Qb[i - 1] = 1.7 * Bo[i - 1] * (Hw[i - 1] - Z[i - 1]) ** 1.5 + 1.3 * np.tan(at[i]) * (Hw[i - 1] - Z[i - 1]) ** 2.5
    # Qb[i - 1] = 3 * Bo[i - 1] * (Hw[i - 1] - Z[i - 1]) ** 1.5 + 1.8 * np.tan(at[i]) * (Hw[i - 1] - Z[i - 1]) ** 2.5
    # 断面平均流速 U
    U[i - 1] = Qb[i - 1] / AA[i - 1] if AA[i - 1] != 0 else np.nan

    # 水位变化量 dl.H
    dlH = (t[i] - t[i - 1]) / Sa[i - 1] * (Qin[i - 1] - Qb[i - 1])

    # 水库水位变化 Hw
    Hw[i] = Hw[i - 1] + dlH

    # 水库面积 Sa
    Sa[i] = (fica * ((Hw[i] + Db) ** 2) - ficb * (Hw[i] + Db) + ficc)

    # 坝顶高程变化相关参数 bt
    if (Z0 - Z_end) != 0:
        bt[i - 1] = bt_0 - ((Z0 - Z[i - 1]) / (Z0 - Z_end)) * (bt_0 - bt_end)
    else:
        bt[i - 1] = bt_0

    if bt[i - 1] < bt_end or bt[i - 1] == bt_end:
        bt[i - 1] = bt_end

    Lx[i - 1] = Z[i - 1] / np.sin(bt[i - 1])

    L[i - 1] = 220 * 3.28 + 2 * (Z0 - Z[i - 1]) / np.tan(bt_0)

    if L[i - 1] <= 0:
        L[i - 1] = Lx[i - 1]

    # 湿周
    if Hk < H1:
        P[i - 1] = Bo[i - 1] + 2 * y[i - 1]
    else:
        if np.cos(at[i]) != 0:
            P[i - 1] = Bo[i - 1] + 2 * y[i - 1] / np.cos(at[i])
        else:
            P[i - 1] = np.inf

    # 输沙率计算参数 S
    if y[i - 1] ** (2 / 3) != 0:
        S[i - 1] = (Mn * U[i - 1] / (y[i - 1] ** (2 / 3))) ** 2
    else:
        S[i - 1] = np.inf

    # 输沙率计算参数 R_
    R_[i - 1] = 1524 * d50 * (y[i - 1] * S[i - 1]) ** 0.5

    # tg_c (临界剪切力参数)
    if R_[i - 1] < 3:
        tg_c = 0.122 / (R_[i - 1] ** 0.97) if R_[i - 1] != 0 else np.inf
    elif 3 <= R_[i - 1] <= 10:
        tg_c = 0.056 / (R_[i - 1] ** 0.266) if R_[i - 1] != 0 else np.inf
    else:  # R_(1.i-1)>10
        tg_c = 0.0205 / (R_[i - 1] ** 0.173) if R_[i - 1] != 0 else np.inf

    # seita
    seita[i - 1] = np.arctan(S[i - 1])

    # a_ (输沙率计算参数)
    a_[i - 1] = np.cos(seita[i - 1]) * (1 - 1.54 * np.tan(seita[i - 1]))

    # tgc
    tgc = a_[i - 1] * tg_c

    # og (无粘性泥沙起动)
    og = 0.0054 * tgc * d50

    # Qs (输沙率)
    Qs[i - 1] = 3.64 * (d90 / d30) ** 0.2 * P[i - 1] * (y[i - 1] ** (2 / 3) / Mn) * (S[i - 1] ** 1.1) * (
                y[i - 1] * S[i - 1] - og)

    # Po (溃口总周长)
    if Hk < H1 and RR[i] == 0:
        Po[i - 1] = Bo[i - 1] + 2 * (Dt - Z[i - 1])
    else:
        if np.cos(at[i]) != 0:
            Po[i - 1] = Bo[i - 1] + 2 * (Dt - Z[i - 1]) / np.cos(at[i])
        else:
            Po[i - 1] = np.inf

    # dl.Hc (高程变化量)
    den_dlhc = (Po[i - 1] * L[i - 1] * (1 - Por))
    dlHc = (t[i] - t[i - 1]) * Qs[i - 1] / den_dlhc

    # Z (溃口高程变化)
    if at[i] != at[i - 1] or dlVV > 0:
        dlV = (Z0 - Z[i - 1]) ** 2 * (np.tan(at[i]) - np.tan(at[i - 1])) * L[i - 1]
        dlVV = dlV - Qs[i - 1] * (t[i] - t[i - 1])
        Z[i] = Z[i - 1]  # 溃口高程不变
    else:
        Z[i] = Z[i - 1] - dlHc  # 溃口高程减小

    if Z[i] < 0:
        Z[i] = 0
    Hm[i] = Dt - Z[i]

# 绘图
min_length_qb = min(len(t), len(Qb))
t_plot_qb = t[:min_length_qb]
Qb_plot = Qb[:min_length_qb]

plt.figure(figsize=(10, 6)) # 可以设置图的大小
plt.plot(t_plot_qb / dt, Qb_plot / 35, linewidth=3.0)
plt.xlabel('time')
plt.ylabel('discharge')
plt.title('discharge line')
plt.grid(True)
plt.show()