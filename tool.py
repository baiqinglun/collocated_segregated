# 放一些小功能的函数

import numpy as np
from fp import Fp

# 计算面积和体积
def calculate_area_volume(dx,dy,dz):
    area_x = dy * dz
    area_y = dx * dz
    area_z = dx * dy
    volume = dx * area_x
    return area_x,area_y,area_z,volume


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
def calculate_face_coefficient(area,dx,ul,ur,mul,mur,rho,sign_f):
    f = rho * Fp(0.5) * (ul + ur) # 与速度有关的项，会导致扩散项发生变化，这里为0
    d = Fp(2.0) * mul * mur / (mul + mur + Fp(1.e-12)) / dx  # 调和平均法计算界面扩散系数 / dx

    # Upwind
    # a = area * (d + max(Fp(0.0),sign_f * f)) # 在无速度时，S * condution_coeff / dx

    # Central
    a = area * d
    return a
