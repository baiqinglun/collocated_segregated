'''
放一些小功能的函数
calculate_area_volume、calculate_face_coefficient、a_pec_pow
'''
from solve import ConvectionScheme
from fp import Fp

def calculate_area_volume(dx, dy, dz):
    '''
    计算面积和体积
    '''
    area_x = dy * dz
    area_y = dx * dz
    area_z = dx * dy
    volume = dx * area_x
    return area_x, area_y, area_z, volume

# 格式
def calculate_face_coefficient(area, dx, ul, ur, mul, mur, rho, sign_f, scheme):
    '''
    dx -------> x单元大小
    ul -------> 左边速度
    ur -------> 右边速度
    mul ------> 左边扩散系数
    mur ------> 右边扩散系数
    rho ------> 密度
    sign_f ---> -1 或者 1
    '''
    f = rho * Fp(0.5) * (ul + ur)
    d = Fp(2.0) * mul * mur / (mul + mur + Fp(1.e-12)) / dx
    a = None
    if scheme == ConvectionScheme.UPWIND:
        # Upwind
        a = area * (d + max(Fp(0.0), sign_f * f))
    elif scheme == ConvectionScheme.CD:
        # Central Difference
        a = area * (d * (Fp(1.0) - Fp(0.5) * abs(f / d)) + max(Fp(0.0), sign_f * f))
    elif scheme == ConvectionScheme.POWER_LOW:
        # Power-law
        a = area * (d * a_pec_pow(abs(f / d)) + max(Fp(0.0), sign_f * f))
    elif scheme == ConvectionScheme.SOU:
        print("ConvectionScheme.sou")
    return a


def a_pec_pow(pec):
    '''
    
    '''
    ap = Fp(1.0) - Fp(0.1) * abs(pec)
    ap = max(Fp(0.0), ap ** 5)

    return ap
