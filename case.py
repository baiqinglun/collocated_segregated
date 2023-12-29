'''存储场数据'''
from enum import Enum
import numpy as np
from fp import Fp
from mesh import MeshManager

class MeshCoefficient2D(Enum):
    '''方程的系数ID(2D)'''
    AP_ID = 0
    AE_ID = 1
    AW_ID = 2
    AN_ID = 3
    AS_ID = 4
    ABRC_ID = 5
    COUNT = 6

class MeshCoefficient3D(Enum):
    '''方程的系数ID(3D)'''
    AP_ID = 0
    AE_ID = 1
    AW_ID = 2
    AN_ID = 3
    AS_ID = 4
    AT_ID = 5
    AB_ID = 6
    ABRC_ID = 7
    COUNT = 8

class CaseManager:
    """存储所有场内的数据"""
    def __init__(self, mesh: MeshManager):
        self.dim = mesh.dim
        self.n_x_cell = mesh.n_x_cell
        self.n_y_cell = mesh.n_y_cell
        self.n_z_cell = mesh.n_z_cell
        self.n_x_point = mesh.n_x_point
        self.n_y_point = mesh.n_y_point
        self.n_z_point = mesh.n_z_point
        self.mesh = mesh

        self.mesh_coefficient = None
        self.n_coefficient = None
        self.u_coefficient = None
        self.v_coefficient = None
        self.w_coefficient = None
        self.t_coefficient = None
        self.p_coefficient = None

        self.initial_u = Fp(0.0)
        self.initial_v = Fp(0.0)
        self.initial_w = Fp(0.0)
        self.initial_uf = Fp(0.0)
        self.initial_vf = Fp(1.0)
        self.initial_wf = Fp(0.0)
        self.initial_t = None
        self.initial_p = None

        self.old_u = None
        self.old_v = None
        self.old_w = None
        self.old_t = None
        self.old_p = None

        self.u = None
        self.v = None
        self.w = None
        self.t = None
        self.p = None

        self.uf = None
        self.vf = None
        self.wf = None
        self.pp = None

        self.ru = None
        self.rv = None
        self.rw = None
        self.rp = None
        self.rt = None

    def create_mesh_data(self):
        """创建速度场、温度场和压力场数据"""
        # 存储场变量
        self.u = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.v = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.w = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.t = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.p = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

        # 存储旧的场变量
        self.old_u = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.old_v = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.old_w = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.old_t = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.old_p = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

        # 使用压力修正方程时的中间变量
        self.initial_uf = Fp(0.0)
        self.initial_vf = Fp(1.0)
        self.initial_wf = Fp(0.0)
        self.uf = np.zeros((self.n_x_point, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.vf = np.zeros((self.n_x_cell, self.n_y_point, self.n_z_cell), dtype=Fp)
        self.wf = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_point), dtype=Fp)
        self.pp = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

        # 源项有关的变量
        self.ru = np.zeros((self.n_x_point, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.rv = np.zeros((self.n_x_cell, self.n_y_point, self.n_z_cell), dtype=Fp)
        self.rw = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_point), dtype=Fp)
        self.rp = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_point), dtype=Fp)
        self.rt = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_point), dtype=Fp)

        # 设置重力
        self.gravity = np.zeros(self.dim,dtype=Fp)

    def create_mesh_coefficient(self):
        """创建每个网格的系数"""
        #mesh_coefficient【含义】:存储系数的枚举id。【类型】Enum。
        self.mesh_coefficient = MeshCoefficient2D if self.dim == 2 else MeshCoefficient3D
        mesh_coefficient_count = self.mesh_coefficient.COUNT.value

        # u_coefficient【含义】存储每个场、每个单元、每个面的的系数值。【类型】Fp
        self.u_coefficient = np.zeros(
            (self.n_x_cell, self.n_y_cell, self.n_z_cell, mesh_coefficient_count), dtype=Fp)
        self.v_coefficient = np.zeros(
            (self.n_x_cell, self.n_y_cell, self.n_z_cell, mesh_coefficient_count), dtype=Fp)
        self.w_coefficient = np.zeros(
            (self.n_x_cell, self.n_y_cell, self.n_z_cell, mesh_coefficient_count), dtype=Fp)
        self.t_coefficient = np.zeros(
            (self.n_x_cell, self.n_y_cell, self.n_z_cell, mesh_coefficient_count), dtype=Fp)
        self.p_coefficient = np.zeros(
            (self.n_x_cell, self.n_y_cell, self.n_z_cell, mesh_coefficient_count), dtype=Fp)

    def set_u(self,u):
        """设置x速度分量"""
        self.initial_u = u
        self.u = u * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_v(self,v):
        """设置y速度分量"""
        self.initial_v = v
        self.v = v * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_w(self,w):
        """设置z速度分量"""
        self.initial_w = w
        self.w = w * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_p(self,p):
        """设置压力"""
        self.initial_p = p
        self.p = p * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_t(self, t):
        """设置温度"""
        self.initial_t = t
        self.t = t * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_uf(self,uf):
        """设置中间变量x速度分量"""
        self.initial_uf = uf
        self.uf = uf * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_vf(self,vf):
        """设置中间变量y速度分量"""
        self.initial_vf = vf
        self.vf = vf * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_wf(self,wf):
        """设置中间变量z速度分量"""
        self.initial_wf = wf
        self.wf = wf * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def get_uf(self):
        '''获取初始化中间变量uf'''
        return self.initial_uf

    def get_vf(self):
        '''获取初始化中间变量vf'''
        return self.initial_vf

    def get_wf(self):
        '''获取初始化中间变量wf'''
        return self.initial_wf
