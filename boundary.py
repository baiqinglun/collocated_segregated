'''边界条件设置'''
from typing import List
import numpy as np
from fp import Fp
from mesh import MeshManager
from boundary_id import FaceId2D, FaceId3D, PhysicsBoundaryID, \
                        TemperatureBoundaryID, BoundaryLimitID,\
                        VelocityBoundaryID

class PhysicsBoundaryCondition:
    '''物理边界条件'''
    def __init__(self):
        self.type = PhysicsBoundaryID.NONE

class VelocityBoundaryCondition:
    '''速度边界条件'''
    def __init__(self):
        self.u_type = VelocityBoundaryID.NONE
        self.v_type = VelocityBoundaryID.NONE
        self.w_type = VelocityBoundaryID.NONE
        self.u = Fp(0.0)
        self.v = Fp(0.0)
        self.w = Fp(0.0)
        self.u_flux = Fp(0.0)
        self.v_flux = Fp(0.0)
        self.w_flux = Fp(0.0)

class TemperatureBoundaryCondition:
    '''温度边界条件'''
    def __init__(self):
        self.type = TemperatureBoundaryID.NONE
        self.t = Fp(0.0)
        self.heat_flux = Fp(0.0)

class FluidBoundaryCondition:
    """
    @name: FluidBoundaryCondition
    @description: 定义流体的边界条件
    @variable:
        face_boundary:每个网格的面
        physics_boundary_condition:[] ——> 物理边界条件
        velocity_boundary_condition:[] ——> 速度边界条件
        temperature_boundary_condition:[] ——> 温度边界条件
    @function: 
        create_face_boundary() ——> 创建边界
        create_boundary() ——> 创建物理边界
        create_boundary_temperature() ——> 创建温度边界
        create_boundary_velocity() ——> 创建速度边界
    """
    def __init__(self, dim):
        self.face_boundary = None  # 存储网格的每个面的边界，判断是边界还是内部
        self.face_id = FaceId2D if dim == 2 else FaceId3D

        self.physics_boundary_condition = []
        self.velocity_boundary_condition = []
        self.temperature_boundary_condition = []
        for _ in range(BoundaryLimitID.COUNT.value):
            self.physics_boundary_condition.append(PhysicsBoundaryCondition())
            self.velocity_boundary_condition.append(VelocityBoundaryCondition())
            self.temperature_boundary_condition.append(TemperatureBoundaryCondition())

        self.p_outlet  = Fp(0.0)
        self.pp_outlet = Fp(0.0)

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
        self.face_boundary:np.ndarray = np.full((n_x_cell, n_y_cell, n_z_cell, 2 * dim),\
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

    def create_boundary_velocity(self, dim,
                                    input_vel_type_xmin, input_xmin_value,
                                    input_vel_type_xmax, input_xmax_value,
                                    input_vel_type_ymin, input_ymin_value,
                                    input_vel_type_ymax, input_ymax_value,
                                    input_vel_type_zmin=["constant","constant","constant"], input_zmin_value=[Fp(0.0), Fp(0.0), Fp(0.0)],
                                    input_vel_type_zmax=["constant","constant","constant"], input_zmax_value=[Fp(0.0), Fp(0.0), Fp(0.0)]):
        '''总网格边界赋速度边界条件'''
        index = BoundaryLimitID.NONE.value
        self.velocity_boundary_condition[index].u_type = VelocityBoundaryID.NONE
        self.velocity_boundary_condition[index].v_type = VelocityBoundaryID.NONE
        self.velocity_boundary_condition[index].u = Fp(0.0)
        self.velocity_boundary_condition[index].v = Fp(0.0)
        self.velocity_boundary_condition[index].u_flux = Fp(0.0)
        self.velocity_boundary_condition[index].v_flux = Fp(0.0)
        self.set_velocity_boundary_condition(BoundaryLimitID.X_MIN.value, input_vel_type_xmin, input_xmin_value)
        self.set_velocity_boundary_condition(BoundaryLimitID.X_MAX.value, input_vel_type_xmax, input_xmax_value)
        self.set_velocity_boundary_condition(BoundaryLimitID.Y_MIN.value, input_vel_type_ymin, input_ymin_value)
        self.set_velocity_boundary_condition(BoundaryLimitID.Y_MAX.value, input_vel_type_ymax, input_ymax_value)

        if dim == 3:
            self.velocity_boundary_condition[index].w_type = VelocityBoundaryID.NONE
            self.velocity_boundary_condition[index].w = Fp(0.0)
            self.velocity_boundary_condition[index].w_flux = Fp(0.0)
            self.set_velocity_boundary_condition(BoundaryLimitID.Z_MIN.value, input_vel_type_zmin, input_zmin_value)
            self.set_velocity_boundary_condition(BoundaryLimitID.Z_MAX.value, input_vel_type_zmax, input_zmax_value)

    def set_velocity_boundary_condition(self, index, buff, value):
        '''设置速度边界条件'''
        attributes = ['u', 'v', 'w']
        for i, attr in enumerate(attributes):
            if buff[i] == 'constant':
                setattr(self.velocity_boundary_condition[index], f'{attr}_type', VelocityBoundaryID.CONSTANT)
                setattr(self.velocity_boundary_condition[index], attr, value[i])
            elif buff[i] == 'flux':
                setattr(self.velocity_boundary_condition[index], f'{attr}_type', VelocityBoundaryID.VELOCITY_FLUX)
                setattr(self.velocity_boundary_condition[index], f'{attr}_flux', value[i])
        
        # setattr函数设置动态属性的值
                
        # if buff[0] == 'constant':
        #     self.velocity_boundary_condition[index].u_type = VelocityBoundaryID.CONSTANT
        #     self.velocity_boundary_condition[index].u = value[0]
        # elif buff[0] == 'flux':
        #     self.velocity_boundary_condition[index].u_type = VelocityBoundaryID.VELOCITY_FLUX
        #     self.velocity_boundary_condition[index].u_flux = value[0]

        # if buff[1] == 'constant':
        #     self.velocity_boundary_condition[index].v_type = VelocityBoundaryID.CONSTANT
        #     self.velocity_boundary_condition[index].v = value[1]
        # elif buff[1] == 'flux':
        #     self.velocity_boundary_condition[index].v_type = VelocityBoundaryID.VELOCITY_FLUX
        #     self.velocity_boundary_condition[index].v_flux = value[1]

        # if buff[2] == 'constant':
        #     self.velocity_boundary_condition[index].w_type = VelocityBoundaryID.CONSTANT
        #     self.velocity_boundary_condition[index].w = value[2]
        # elif buff[2] == 'flux':
        #     self.velocity_boundary_condition[index].w_type = VelocityBoundaryID.VELOCITY_FLUX
        #     self.velocity_boundary_condition[index].w_flux = value[2]

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
