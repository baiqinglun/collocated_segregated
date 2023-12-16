'''边界条件设置'''
import numpy as np
from fp import Fp
from mesh import MeshManager
from boundary_id import FaceId2D, FaceId3D, PhysicsBoundaryID, \
                        TemperatureBoundaryID, BoundaryLimitID

class PhysicsBoundaryCondition:
    '''物理边界条件'''
    def __init__(self):
        self.type = 0


class TemperatureBoundaryCondition:
    '''温度边界条件'''
    def __init__(self):
        self.type = 0
        self.t = Fp(0.0)
        self.heat_flux = Fp(0.0)

class FluidBoundaryCondition:
    """
    @name: FluidBoundaryCondition
    @description: 定义流体的边界条件
    @variable:
        face_boundary:每个网格的面
        physics_boundary_condition:[]：物理边界条件
        temperature_boundary_condition:[]：温度边界条件
    @function: 
        create_face_boundary()：创建边界
        create_boundary()：创建物理边界
        create_boundary_temperature()：创建温度边界
    """
    def __init__(self, dim):
        self.face_boundary = None  # 存储网格的每个面的边界，判断是边界还是内部
        self.face_id = FaceId2D if dim == 2 else FaceId3D

        self.physics_boundary_condition = []
        self.temperature_boundary_condition = []
        for _ in range(BoundaryLimitID.COUNT.value):
            self.physics_boundary_condition.append(PhysicsBoundaryCondition())
            self.temperature_boundary_condition.append(TemperatureBoundaryCondition())

    def create_face_boundary(self, mesh: MeshManager):
        '''创建单元面边界条件'''
        dim = mesh.dim
        n_x_cell = mesh.n_x_cell
        n_y_cell = mesh.n_y_cell
        n_z_cell = mesh.n_z_cell
        x_min = mesh.coordinate_range[0]
        x_max = mesh.coordinate_range[1]
        y_min = mesh.coordinate_range[2]
        y_max = mesh.coordinate_range[3]
        z_min = mesh.coordinate_range[4]
        z_max = mesh.coordinate_range[5]
        x = mesh.x
        y = mesh.y
        z = mesh.z

        # 对应Boundary的枚举类型
        self.face_boundary = np.full((n_x_cell, n_y_cell, n_z_cell, 2 * dim),\
                                     BoundaryLimitID.NONE, dtype=BoundaryLimitID)

        # 确定单元的每个面的位置，东西南北上下
        eps = Fp(1.e-12)  # 确定面或者是否为单元的边界
        for k in range(n_z_cell):
            for j in range(n_y_cell):
                for i in range(n_x_cell):
                    x0 = x[i]
                    if abs(x0 - x_min) < eps:
                        self.face_boundary[i, j, k, self.face_id.WEST.value] = BoundaryLimitID.X_MIN

                    x0 = x[i + 1]
                    if abs(x0 - x_max) < eps:
                        self.face_boundary[i, j, k, self.face_id.EAST.value] = BoundaryLimitID.X_MAX

                    y0 = y[j]
                    if abs(y0 - y_min) < eps:
                        self.face_boundary[i, j, k, self.face_id.SOUTH.value] = BoundaryLimitID.Y_MIN

                    y0 = y[j + 1]
                    if abs(y0 - y_max) < eps:
                        self.face_boundary[i, j, k, self.face_id.NORTH.value] = BoundaryLimitID.Y_MAX

                    if dim == 3:
                        z0 = z[k]
                        if abs(z0 - z_min) < eps:
                            self.face_boundary[i, j, k, self.face_id.BOTTOM.value] = BoundaryLimitID.Z_MIN

                        z0 = z[k + 1]
                        if abs(z0 - z_max) < eps:
                            self.face_boundary[i, j, k, self.face_id.TOP.value] = BoundaryLimitID.Z_MAX

    def create_boundary(self, dim, input_bc_xmin, input_bc_xmax, input_bc_ymin, input_bc_ymax,
                        input_bc_zmin=PhysicsBoundaryID.NONE, input_bc_zmax=PhysicsBoundaryID.NONE):
        '''创建边界条件'''
        index = BoundaryLimitID.NONE.value
        self.physics_boundary_condition[index].type = PhysicsBoundaryID.NONE

        self.set_physics_boundary_condition(BoundaryLimitID.X_MIN.value, input_bc_xmin)
        self.set_physics_boundary_condition(BoundaryLimitID.X_MAX.value, input_bc_xmax)
        self.set_physics_boundary_condition(BoundaryLimitID.Y_MIN.value, input_bc_ymin)
        self.set_physics_boundary_condition(BoundaryLimitID.Y_MAX.value, input_bc_ymax)
        if dim == 3:
            self.set_physics_boundary_condition(BoundaryLimitID.Z_MIN.value, input_bc_zmin)
            self.set_physics_boundary_condition(BoundaryLimitID.Z_MAX.value, input_bc_zmax)

    def set_physics_boundary_condition(self, index, buff):
        '''设置物理边界条件'''
        if buff == 'inlet':
            self.physics_boundary_condition[index].type = PhysicsBoundaryID.INLET
        elif buff == 'outlet':
            self.physics_boundary_condition[index].type = PhysicsBoundaryID.OUTLET
        elif buff == 'wall':
            self.physics_boundary_condition[index].type = PhysicsBoundaryID.WALL
        else:
            self.physics_boundary_condition[index].type = PhysicsBoundaryID.NONE

    def create_boundary_temperature(self, dim,
                                    input_temp_type_xmin, input_xmin_value,
                                    input_temp_type_xmax, input_xmax_value,
                                    input_temp_type_ymin, input_ymin_value,
                                    input_temp_type_ymax, input_ymax_value,
                                    input_temp_type_zmin="constant", input_zmin_value="373",
                                    input_temp_type_zmax="constant", input_zmax_value="373"):
        '''总网格边界赋温度边界条件'''
        index = BoundaryLimitID.NONE.value
        self.temperature_boundary_condition[index].t = Fp(0.0)
        self.temperature_boundary_condition[index].heat_flux = Fp(0.0)
        self.temperature_boundary_condition[index].type = PhysicsBoundaryID.NONE
        self.set_temperature_boundary_condition(BoundaryLimitID.X_MIN.value, input_temp_type_xmin, input_xmin_value)
        self.set_temperature_boundary_condition(BoundaryLimitID.X_MAX.value, input_temp_type_xmax, input_xmax_value)
        self.set_temperature_boundary_condition(BoundaryLimitID.Y_MIN.value, input_temp_type_ymin, input_ymin_value)
        self.set_temperature_boundary_condition(BoundaryLimitID.Y_MAX.value, input_temp_type_ymax, input_ymax_value)
        if dim == 3:
            self.set_temperature_boundary_condition(BoundaryLimitID.Z_MIN.value, input_temp_type_zmin, input_zmin_value)
            self.set_temperature_boundary_condition(BoundaryLimitID.Z_MAX.value, input_temp_type_zmax, input_zmax_value)

    def set_temperature_boundary_condition(self, index, buff, value):
        '''设置温度边界条件'''
        if buff == 'constant':
            self.temperature_boundary_condition[index].type = TemperatureBoundaryID.CONSTANT
            self.temperature_boundary_condition[index].t = value
        elif buff == 'heat_flux':
            self.temperature_boundary_condition[index].type = TemperatureBoundaryID.HEAT_FLUX
            self.temperature_boundary_condition[index].heat_flux = value
