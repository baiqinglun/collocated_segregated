'''网格信息'''
import numpy as np
from fp import Fp

class MeshManager:
    """
    @name：MeshManager
    @description: 定义网格的相关信息
    @variable:
        x_cell_centroid[]：存储知心坐标
        coordinate_range[]：坐标轴范围
        dx：单元x轴向大小
        n_x_cell：x轴向单元个数
        n_x_point：x轴向坐标点个数
    @function: 
        create_coordinates()：创建坐标轴
    """
    def __init__(self, dim, n_x_cell, n_y_cell, n_z_cell=1):
        self.z_cell_centroid = None
        self.y_cell_centroid = None
        self.x_cell_centroid = None
        self.coordinate_range = None
        self.dz = None
        self.dy = None
        self.dx = None
        self.z = None
        self.y = None
        self.x = None
        self.dim = dim
        self.n_x_cell = n_x_cell
        self.n_y_cell = n_y_cell
        self.n_z_cell = 1 if self.dim == 2 else n_z_cell
        self.n_x_point = self.n_x_cell + 1
        self.n_y_point = self.n_y_cell + 1
        self.n_z_point = self.n_z_cell + 1

    def create_coordinates(self, coordinate_range):
        '''创建坐标轴'''
        self.coordinate_range = coordinate_range
        self.dx = (self.coordinate_range[1] - self.coordinate_range[0]) / self.n_x_cell
        self.dy = (self.coordinate_range[3] - self.coordinate_range[2]) / self.n_y_cell
        self.dz = (self.coordinate_range[5] - self.coordinate_range[4]) / self.n_z_cell

        # 坐标轴
        self.x = np.zeros(self.n_x_point, dtype=Fp)
        self.y = np.zeros(self.n_y_point, dtype=Fp)
        self.z = np.zeros(self.n_z_point, dtype=Fp)
        for i in range(self.n_x_point):
            self.x[i] = self.coordinate_range[0] + self.dx * Fp(i)
        for i in range(self.n_y_point):
            self.y[i] = self.coordinate_range[2] + self.dy * Fp(i)
        for i in range(self.n_z_point):
            self.z[i] = self.coordinate_range[4] + self.dz * Fp(i)

        # 坐标轴质心坐标
        self.x_cell_centroid = np.zeros(self.n_x_cell, dtype=Fp)
        self.y_cell_centroid = np.zeros(self.n_y_cell,dtype=Fp)
        self.z_cell_centroid = np.zeros(self.n_z_cell,dtype=Fp)
        for i in range(self.n_x_cell):
            self.x_cell_centroid[i] = 0.5 * (self.x[i] + self.x[i+1])
        for i in range(self.n_y_cell):
            self.y_cell_centroid[i] = 0.5 * (self.y[i] + self.y[i+1])
        for i in range(self.n_z_cell):
            self.z_cell_centroid[i] = 0.5 * (self.z[i] + self.z[i+1])
