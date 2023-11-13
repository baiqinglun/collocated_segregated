# 放一些小功能的函数

from fp import Fp
from solve import EquationType, ConvectionScheme


# 计算面积和体积
def calculate_area_volume(dx, dy, dz):
    area_x = dy * dz
    area_y = dx * dz
    area_z = dx * dy
    volume = dx * area_x
    return area_x, area_y, area_z, volume


'''
dx     ：x单元大小
ul     ：左边速度
ur     ：右边速度
mul     ：左边扩散系数
mur     ：右边扩散系数
rho    ：密度
sign_f ：-1 或者 1
'''


# 格式
def calculate_face_coefficient(area, dx, ul, ur, mul, mur, rho, sign_f, scheme):
    f = rho * Fp(0.5) * (ul + ur)
    d = Fp(2.0) * mul * mur / (mul + mur + Fp(1.e-12)) / dx
    a = None
    if scheme == ConvectionScheme.upwind:
        # Upwind
        a = area * (d + max(Fp(0.0), sign_f * f))
    elif scheme == ConvectionScheme.cd:
        # Central Difference
        a = area * (d * (Fp(1.0) - Fp(0.5) * abs(f / d)) + max(Fp(0.0), sign_f * f))
    elif scheme == ConvectionScheme.power_law:
        # Power-law
        a = area * (d * a_pec_pow(abs(f / d)) + max(Fp(0.0), sign_f * f))
    elif scheme == ConvectionScheme.sou:
        print("ConvectionScheme.sou")
    return a


def a_pec_pow(pec):
    # Incoming variable
    # pec: float

    ap = Fp(1.0) - Fp(0.1) * abs(pec)
    ap = max(Fp(0.0), ap ** 5)

    return ap
