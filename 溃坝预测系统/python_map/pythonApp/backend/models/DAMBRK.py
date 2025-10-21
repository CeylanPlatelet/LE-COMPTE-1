import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# 读取 Excel 文件中的参数
params_file = 'myFloodRoutingSys/dam/input/params3.xlsx'
params = pd.read_excel(params_file, sheet_name=0, header=None)
params2 = pd.read_excel(params_file, sheet_name=1, header=None)
# 提取参数
step = params.iloc[28, 1]  # 时间步长
Hs = params.iloc[1, 1]  # 初始总高m(坝体顶部高程)
QIN = params2.values[:, 0]  # 入流量
Hw0 = params.iloc[3, 1]  # 湖水水位(初始水位高程)
d50 = params.iloc[9, 1:5].values.flatten()
MI = d50[1] ** (1/6) / 12  # 曼宁粗糙系数
Db = params.iloc[16, 1]  # 基岩高度m(坝体底部高程)
L_1 = params.iloc[17, 1]  # 坝体底部宽度（顺河长）
L0 = params.iloc[18, 1]  # 坝体顶部宽度
L_2 = params.iloc[43, 1]  # 坝体顶部宽度
n = int(params.iloc[19, 1] / step / 2)  # 循环次数
fica = params.iloc[21, 1]  # 方程中变量平方前系数
ficb = params.iloc[22, 1]  # 方程中变量前系数
ficc = params.iloc[23, 1]  # 方程中常数
stage1 = params.iloc[35, 1]  # 堰塞坝材料分层一阶段
stage2 = params.iloc[36, 1]  # 堰塞坝材料分层二阶段
stage3 = params.iloc[37, 1]  # 堰塞坝材料分层三阶段
BH = params.iloc[27, 1]  # 河道宽度
V0 = params.iloc[42, 1]  # 初始库容
dt = params.iloc[20, 1]
bt_0 = params.iloc[11, 1]  # 背水坡坡角
ps = params.iloc[32, 1:5].values.flatten()  # 颗粒容重kg/m^3
FAI = params.iloc[33, 1:5].values.flatten()  # 内摩擦角
c = params.iloc[25, 1]  # 粘聚力
m = params.iloc[30, 1]  # 转换系数
state = params.iloc[49, 1]
Z0 = params.iloc[4, 1]  # 溃口底部标高
D50 = params.iloc[9, 1:5].values.flatten()
d90 = params.iloc[10, 1]
pw = params.iloc[31, 1]  # 水容重kg/m^3
B0 = params.iloc[7, 1]  # 溃口底部宽度(溃口宽度)
Cd = params.iloc[0, 1]  # 流量系数m^0.5/s
ERP = params.iloc[45, 1]  # 侵蚀度
g = 9.81

# 初始化变量
Hm0 = Hs - Z0
Hm = np.ones(n) * Hm0
As0 = (fica * (Hw0 + Db) ** 2 - ficb * (Hw0 + Db) + ficc)
V0 = params.iloc[42, 1]  # 初始库容

# 计算溃口底部最终高程和宽度
if ERP == 3 or ERP == 4:
    erp1 = -0.322  # 高侵蚀
    erp2 = 1.360   # 高侵蚀
    erp3 = -0.602  # 高侵蚀
else:
    erp1 = -0.709  # 低侵蚀
    erp2 = 0.710   # 低侵蚀
    erp3 = -0.662  # 低侵蚀

Zbm = Hs ** 0.875 * V0 ** 0.016 * np.exp(erp1)  # 溃口底部最终高程
b_end = -0.007 * Hs ** 2 + 0.053 * V0 ** (1/3) + erp2 * Hs  # 最终溃口宽度
tao = Hs ** (-0.425) * V0 ** 0.236 * np.exp(erp3)

# 初始化数组
ht = np.zeros(n)
Z = np.ones(n) * Z0
Hw = np.ones(n) * Hw0
As = np.ones(n) * As0
V = np.ones(n) * V0
t = np.zeros(n)
Qb = np.zeros(n)  # 溃口流量
B = np.zeros(n)
N = np.zeros(n, dtype=int)
Qin = np.zeros(n)
H0 = np.zeros(n)
tb = np.zeros(n)
cv = np.zeros(n)

# 主循环
for i in range(1, n):
    t[i] = dt * step * i
    if state == 888:
        if Hm[i-1] < stage1:
            fai = FAI[0]
        elif stage1 <= Hm[i-1] < stage2:
            fai = FAI[1]
        elif stage2 <= Hm[i-1] < stage3:
            fai = FAI[2]
        else:
            fai = FAI[3]
    else:
        fai = FAI[0]

    N[i-1] = round((t[i] + dt) / dt)
    Qin[i] = QIN[int(N[i-1]) - 1] + (QIN[int(N[i-1])] - QIN[int(N[i-1]) - 1]) * \
                (t[i] / dt - step * (i + 1)) / (step * (i + 2) - t[i] / dt)

    tb[i] = t[i] / dt  # 溃决时间

    H0[i] = (Qin[i] - Qb[i-1]) / As[i-1] * (t[i] - t[i-1])  # 湖面高程变化量
    Hw[i] = Hw[i-1] + H0[i]

    As[i] = (fica * (Hw[i] + Db) ** 2 - ficb * (Hw[i] + Db) + ficc)  # 湖面面积

    V[i] = V[i-1] - As[i] * (Hw[i-1] - Hw[i]) * (t[i] - t[i-1]) / dt

    if V[i] < 0:
        V[i] = 0

    Z[i] = Z0 - (Z0 - Zbm) * (tb[i] / tao)  # 溃口底部高程
    if Z[i] <= Zbm:
        Z[i] = Zbm

    Hm[i-1] = Hs - Z[i]

    B[i] = b_end * (tb[i] / tao)  # 溃口瞬时底部宽度
    if B[i] >= b_end:
        B[i] = b_end

    cv[i] = 1.0 + (0.023 * Qin[i] ** 2) / (L_1 ** 2 * (Hw[i] - Zbm) ** 2 * (Hw[i] - Z[i]))

    z = np.tan(np.pi / 4 + fai / 2)  # 溃口边坡系数

    Qb[i] = cv[i] * m * (3.1 * B[i] * (Hw[i] - Z[i]) ** 1.5 + 2.45 * z * (Hw[i] - Z[i]) ** 2.5)  # 宽顶堰流公式

    if Qb[i] < 0:
        Qb[i] = 0

# 绘图
plt.figure()
plt.plot(t / 3600, Qb, linewidth=3.0)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('discharge')
plt.title('discharge line')
plt.show()