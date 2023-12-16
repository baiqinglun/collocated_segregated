'''存储场数据'''
from enum import Enum
import numpy as np
from fp import Fp
from mesh import MeshManager

class MeshCoefficient2D(Enum):
    '''
    方程的系数ID(2D)
    '''
    AP_ID = 0
    AW_ID = 1
    AE_ID = 2
    AS_ID = 3
    AN_ID = 4
    ABRC_ID = 5
    COUNT = 6

class MeshCoefficient3D(Enum):
    '''
    方程的系数ID(3D)
    '''
    AP_ID = 0
    AW_ID = 1
    AE_ID = 2
    AS_ID = 3
    AN_ID = 4
    AT_ID = 5
    AB_ID = 6
    ABRC_ID = 7
    COUNT = 8

class CaseManager:
    """
    存储所有场内的数据
    """
    def __init__(self, mesh: MeshManager):
        self.dim = mesh.dim
        self.n_x_cell = mesh.n_x_cell
        self.n_y_cell = mesh.n_y_cell
        self.n_z_cell = mesh.n_z_cell
        self.mesh = mesh

        self.mesh_coefficient = None
        self.n_coefficient = None
        self.u_coefficient = None
        self.v_coefficient = None
        self.w_coefficient = None
        self.t_coefficient = None
        self.p_coefficient = None

        self.initial_u = None
        self.initial_v = None
        self.initial_w = None
        self.initial_uf = None
        self.initial_vf = None
        self.initial_wf = None
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

    def create_mesh_data(self):
        """
        创建速度场、温度场和压力场数据
        """
        self.old_u = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.old_v = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.old_w = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.old_t = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.old_p = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

        self.u = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.v = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.w = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.t = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.p = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

        # 使用压力修正方程时的中间变量
        self.uf = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.vf = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.wf = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.pp = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

        # 源项有关的变量
        self.ru = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.rv = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.rw = np.zeros((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_temperature(self, temperature):
        """
        设置温度
        """
        self.t = temperature * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def create_mesh_coefficient(self):
        """
        创建每个网格的系数
        """
        self.mesh_coefficient = MeshCoefficient2D if self.dim == 2 else MeshCoefficient3D
        mesh_coefficient_count = self.mesh_coefficient.COUNT.value

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
        """
        设置x速度分量
        """
        self.initial_u = u
        self.u = u * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_v(self,v):
        """
        设置y速度分量
        """
        self.initial_v = v
        self.v = v * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_w(self,w):
        """
        设置z速度分量
        """
        self.initial_w = w
        self.w = w * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)

    def set_p(self,p):
        """
        设置压力
        """
        self.initial_p = p
        self.p = p * np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
