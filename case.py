import numpy as np
from fp import Fp
from mesh import MeshManager
from enum import Enum

'''
方程的系数id
分别定义二维和三维方程系数id的枚举类型，
'''
class MeshCoefficient2D(Enum):
    aP_id = 0
    aW_id = 1
    aE_id = 2
    aS_id = 3
    aN_id = 4
    ab_id = 5
    count = 6
class MeshCoefficient3D(Enum):
    aP_id = 0
    aW_id = 1
    aE_id = 2
    aS_id = 3
    aN_id = 4
    aT_id = 5
    aB_id = 6
    ab_id = 7
    count = 8


class CaseManager:
    def __init__(self, mesh: MeshManager):
        self.t_coefficient = None
        self.mesh_coefficient = None
        self.n_coefficient = None
        self.u = None
        self.w = None
        self.v = None
        self.old_t = None
        self.t = None
        self.mesh = mesh

    # 创建温度场和速度场数据
    def create_mesh_data(self):
        self.t = np.zeros((self.mesh.n_x_cell, self.mesh.n_y_cell, self.mesh.n_z_cell), dtype=Fp)
        self.old_t = np.zeros((self.mesh.n_x_cell, self.mesh.n_y_cell, self.mesh.n_z_cell), dtype=Fp)
        self.u = np.zeros((self.mesh.n_x_cell, self.mesh.n_y_cell, self.mesh.n_z_cell), dtype=Fp)
        self.v = np.zeros((self.mesh.n_x_cell, self.mesh.n_y_cell, self.mesh.n_z_cell), dtype=Fp)
        self.w = np.zeros((self.mesh.n_x_cell, self.mesh.n_y_cell, self.mesh.n_z_cell), dtype=Fp)

    # 设置温度
    def set_temperature(self, temperature):
        self.t = temperature * np.ones((self.mesh.n_x_cell, self.mesh.n_y_cell, self.mesh.n_z_cell), dtype=Fp)

    # 创建方程的系数
    def create_mesh_coefficient(self):
        self.mesh_coefficient = MeshCoefficient2D if self.mesh.dim == 2 else MeshCoefficient3D
        self.t_coefficient = np.zeros(
            (self.mesh.n_x_cell, self.mesh.n_y_cell, self.mesh.n_z_cell, self.mesh_coefficient.count.value), dtype=Fp)
