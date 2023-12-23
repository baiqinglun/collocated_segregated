'''分离求解器公共函数'''
from typing import List
import numpy as np
from fp import Fp
from tool import calculate_area_volume, calculate_face_coefficient
from boundary import PhysicsBoundaryCondition, TemperatureBoundaryCondition,\
    FluidBoundaryCondition,VelocityBoundaryCondition
from mesh import MeshManager,Direction
from case import MeshCoefficient2D, MeshCoefficient3D,CaseManager
from solve import SolveManager
from boundary_id import PhysicsBoundaryID, TemperatureBoundaryID,FaceId2D,FaceId3D
from fluid import Fluid



def solve_diffusion_coefficient(dx, dy, dz, t_coefficient, mesh_coefficient, dim,
                                 n_x_cell, n_y_cell, n_z_cell,
                                 dt, specific_heat_capacity, conductivity_coefficient, source_term,
                                 density, t, u, v, w, conv_scheme):
    '''求解热扩散系数'''
    # 获取Cell的面积和体积
    area_x, area_y, area_z, vol = calculate_area_volume(dx, dy, dz)

    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                # 初始化系数
                a_e = a_w = a_s = a_n = a_t = a_b = a_p = Fp(0.0)
                s_c = source_term
                s_p = Fp(0.0)
                b = Fp(0.0)

                # 速度分量
                vuw = {
                    "ul": u[i, j, k],
                    "ur": u[i, j, k],
                    "vl": v[i, j, k],
                    "vr": v[i, j, k],
                    "wl": w[i, j, k],
                    "wr": w[i, j, k]
                }
                # 热传导系数
                mu = {
                    "mul": conductivity_coefficient,
                    "mur": conductivity_coefficient,
                }

                # -----------------------------5.62、5.63计算每个单元每个面的系数--------------------------------#
                # east face
                if i == n_x_cell - 1:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i + 1, j, k])
                a_e = calculate_face_coefficient(area_x, dx, vuw["ul"], vuw["ur"], mu["mul"], mu["mur"],
                                                 rho, -Fp(1.0), conv_scheme)

                # west face
                if i == 0:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i - 1, j, k])
                a_w = calculate_face_coefficient(area_x, dx, vuw["ul"], vuw["ur"], mu["mul"], mu["mur"],
                                                 rho, Fp(1.0), conv_scheme)

                # north face
                if j == n_y_cell - 1:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i, j + 1, k])
                a_n = calculate_face_coefficient(area_y, dy, vuw["vl"], vuw["vr"], mu["mul"], mu["mur"],
                                                 rho, -Fp(1.0), conv_scheme)

                # south face
                if j == 0:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i, j - 1, k])
                a_s = calculate_face_coefficient(area_y, dy, vuw["vl"], vuw["vr"], mu["mul"], mu["mur"],
                                                 rho, Fp(1.0), conv_scheme)

                if dim == 3:
                    # top face
                    if k == n_z_cell - 1:
                        rho = density[i, j, k]
                    else:
                        rho = Fp(0.5) * (density[i, j, k] + density[i, j, k + 1])
                    a_t = calculate_face_coefficient(area_z, dz, vuw["wl"], vuw["wr"], mu["mul"],
                                                     mu["mur"], rho, -Fp(1.0), conv_scheme)

                    # bottom face
                    if k == 0:
                        rho = density[i, j, k]
                    else:
                        rho = Fp(0.5) * (density[i, j, k] + density[i, j, k - 1])
                    a_b = calculate_face_coefficient(area_z, dz, vuw["wl"], vuw["wr"], mu["mul"],
                                                     mu["mur"], rho, Fp(1.0), conv_scheme)
                # 求解ap系数及源项
                rho = density[i, j, k]
                a_p0 = rho * specific_heat_capacity * vol / dt
                a_p = a_e + a_w + a_s + a_n + a_t + a_b + a_p0 - s_p * vol

                # 存储数据
                t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] = a_p
                t_coefficient[i, j, k, mesh_coefficient.AW_ID.value] = -a_w
                t_coefficient[i, j, k, mesh_coefficient.AE_ID.value] = -a_e
                t_coefficient[i, j, k, mesh_coefficient.AS_ID.value] = -a_s
                t_coefficient[i, j, k, mesh_coefficient.AN_ID.value] = -a_n
                if dim == 3:
                    t_coefficient[i, j, k, mesh_coefficient.AT_ID.value] = -a_t
                    t_coefficient[i, j, k, mesh_coefficient.AB_ID.value] = -a_b
                b = s_c * vol + a_p0 * t[i, j, k]
                t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] = b

def modify_diffusion_boundary_coefficient(x_cell_centroid, y_cell_centroid, z_cell_centroid, face_boundary, face_id,
                                            dim, n_x_cell, n_y_cell, n_z_cell,
                                            conductivity_coefficient, mesh_coefficient, x, y, z, t, t_coefficient,
                                            physics_boundary_condition:List[PhysicsBoundaryCondition],
                                            temperature_boundary_condition: List[TemperatureBoundaryCondition]):
    '''考虑温度边界，求解系数'''
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                # 存储每个单元的边界
                faces = []
                enum_face_id_list = list(face_id)
                if dim == 2:
                    enum_face_id_list = enum_face_id_list[:4]
                elif dim == 3:
                    enum_face_id_list = enum_face_id_list[:6]
                for enum_face_id in enum_face_id_list:
                    faces.append(face_boundary[i, j, k, enum_face_id.value])
                # self.face_id = FaceId2D if dim == 2 else FaceId3D
                # 对应Boundary的枚举类型
                # self.face_boundary:np.ndarray = np.full((n_x_cell, n_y_cell, n_z_cell, 2 * dim),\
                                           # BoundaryLimitID.NONE, dtype=BoundaryLimitID)
                # 物理边界类型与值
                face_physics_boundary = []
                face_temperature_boundary = []
                face_temperature = []
                face_flux = []
                for face in faces:
                    face_physics_boundary.append(physics_boundary_condition[face.value].type)
                    face_temperature_boundary.append(temperature_boundary_condition[face.value].type)
                    face_temperature.append(temperature_boundary_condition[face.value].t)
                    face_flux.append(temperature_boundary_condition[face.value].heat_flux)

                # 根据边界条件修改（在原系数基础上修改）每个单元不同面的系数
                all_coefficient_list = [
                    [face_physics_boundary[0], face_temperature_boundary[0], mesh_coefficient.AE_ID.value,
                     face_temperature[0], face_temperature[0], x[i + 1], x_cell_centroid[i]],
                    [face_physics_boundary[1], face_temperature_boundary[1], mesh_coefficient.AW_ID.value,
                     face_temperature[1], face_temperature[1], x[i], x_cell_centroid[i - 1]],
                    [face_physics_boundary[2], face_temperature_boundary[2], mesh_coefficient.AS_ID.value,
                     face_temperature[2], face_temperature[2], y[j + 1], y_cell_centroid[j]],
                    [face_physics_boundary[3], face_temperature_boundary[3], mesh_coefficient.AN_ID.value,
                     face_temperature[3], face_temperature[3], y[j], y_cell_centroid[j - 1]]]
                if dim == 3:
                    all_coefficient_list.append(
                        [face_physics_boundary[4], face_temperature_boundary[4], mesh_coefficient.AT_ID.value,
                         face_temperature[4], face_temperature[4], z[k + 1], z_cell_centroid[k]])
                    all_coefficient_list.append(
                        [face_physics_boundary[5], face_temperature_boundary[5], mesh_coefficient.AB_ID.value,
                         face_temperature[5], face_temperature[5], z[k], z_cell_centroid[k - 1]])
                for value in all_coefficient_list:
                    if value[0] == PhysicsBoundaryID.INLET:
                        t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] -= t_coefficient[i, j, k, value[2]]
                        t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] -= Fp(2.0) * t_coefficient[
                            i, j, k, value[2]] * value[3]
                        t_coefficient[i, j, k, value[2]] = Fp(0.0)
                    elif value[0] == PhysicsBoundaryID.OUTLET:
                        t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] += t_coefficient[i, j, k, value[2]]
                        t_coefficient[i, j, k, value[2]] = Fp(0.0)
                    elif value[0] == PhysicsBoundaryID.WALL:
                        if value[1] == TemperatureBoundaryID.CONSTANT:
                            t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] -= t_coefficient[i, j, k, value[2]]
                            t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] -= Fp(2.0) * t_coefficient[
                                i, j, k, value[2]] * value[3]
                            t_coefficient[i, j, k, value[2]] = Fp(0.0)
                        elif value[1] == TemperatureBoundaryID.HEAT_FLUX:
                            t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] += t_coefficient[i, j, k, value[2]]
                            t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] += t_coefficient[i, j, k, value[2]] * \
                                                                                    value[4] * Fp(2.0) * (
                                                                                                value[5] - value[
                                                                                            6]) / conductivity_coefficient
                            t_coefficient[i, j, k, value[2]] = Fp(0.0)



def solve_conduction_coefficient(dx, dy, dz, t_coefficient, mesh_coefficient, dim,
                                      n_x_cell, n_y_cell, n_z_cell,
                                      dt, specific_heat_capacity, conductivity_coefficient, source_term,
                                      density, t, u, v, w, conv_scheme):
    
    '''求解对流项系数'''
    area_x, area_y, area_z, vol = calculate_area_volume(dx, dy, dz)

    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                # 初始化系数
                a_e = a_w = a_s = a_n = a_t = a_b = a_p = Fp(0.0)
                s_c = source_term
                s_p = Fp(0.0)
                b = Fp(0.0)
                
                # 速度分量 QUESTION:为什么是[i, j, k]
                vuw = {
                    "ul": u[i, j, k],
                    "ur": u[i, j, k],
                    "vl": v[i, j, k],
                    "vr": v[i, j, k],
                    "wl": w[i, j, k],
                    "wr": w[i, j, k]
                }
                # 热传导系数
                mu = {
                    "mul": conductivity_coefficient[i,j,k],
                    "mur": conductivity_coefficient[i,j,k],
                }

                # -----------------------------5.62、5.63计算每个单元每个面的系数--------------------------------#
                # east face
                if i == n_x_cell - 1:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i + 1, j, k])
                a_e = calculate_face_coefficient(conv_scheme,area_x, dx, vuw["ul"], vuw["ur"], mu["mul"], mu["mur"],
                                                 rho, -Fp(1.0))

                # west face
                if i == 0:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i - 1, j, k])
                a_w = calculate_face_coefficient(conv_scheme,area_x, dx, vuw["ul"], vuw["ur"], mu["mul"], mu["mur"],
                                                 rho, Fp(1.0))

                # north face
                if j == n_y_cell - 1:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i, j + 1, k])
                a_n = calculate_face_coefficient(conv_scheme,area_y, dy, vuw["vl"], vuw["vr"], mu["mul"], mu["mur"],
                                                 rho, -Fp(1.0))

                # south face
                if j == 0:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i, j - 1, k])
                a_s = calculate_face_coefficient(conv_scheme,area_y, dy, vuw["vl"], vuw["vr"], mu["mul"], mu["mur"],
                                                 rho, Fp(1.0))

                if dim == 3:
                    # top face
                    if k == n_z_cell - 1:
                        rho = density[i, j, k]
                    else:
                        rho = Fp(0.5) * (density[i, j, k] + density[i, j, k + 1])
                    a_t = calculate_face_coefficient(conv_scheme,area_z, dz, vuw["wl"], vuw["wr"], mu["mul"],
                                                     mu["mur"], rho, -Fp(1.0))

                    # bottom face
                    if k == 0:
                        rho = density[i, j, k]
                    else:
                        rho = Fp(0.5) * (density[i, j, k] + density[i, j, k - 1])
                    a_b = calculate_face_coefficient(conv_scheme,area_z, dz, vuw["wl"], vuw["wr"], mu["mul"],
                                                     mu["mur"], rho, Fp(1.0))
                # 求解ap系数及源项
                rho = density[i, j, k]
                a_p0 = rho * specific_heat_capacity* vol / dt
                a_p = a_e + a_w + a_s + a_n + a_t + a_b + a_p0 - s_p * vol

                # 存储数据
                t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] = a_p
                t_coefficient[i, j, k, mesh_coefficient.AW_ID.value] = -a_w
                t_coefficient[i, j, k, mesh_coefficient.AE_ID.value] = -a_e
                t_coefficient[i, j, k, mesh_coefficient.AS_ID.value] = -a_s
                t_coefficient[i, j, k, mesh_coefficient.AN_ID.value] = -a_n
                if dim == 3:
                    t_coefficient[i, j, k, mesh_coefficient.AT_ID.value] = -a_t
                    t_coefficient[i, j, k, mesh_coefficient.AB_ID.value] = -a_b
                b = s_c * vol + a_p0 * t[i, j, k]
                t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] = b


def modify_conduction_boundary_coefficient(x_cell_centroid, y_cell_centroid, z_cell_centroid,\
                                                     face_boundary, face_id, dim, n_x_cell,\
                                                     n_y_cell, n_z_cell,\
                                                     conductivity_coefficient, mesh_coefficient,\
                                                     x,y,z,t_coefficient,\
                                                     physics_boundary_condition,\
                                                     temperature_boundary_condition):
    '''考虑边界条件，求解对流项系数'''
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                # 存储每个单元的边界
                faces = []
                enum_face_id_list = list(face_id)
                if dim == 2:
                    enum_face_id_list = enum_face_id_list[:4]
                elif dim == 3:
                    enum_face_id_list = enum_face_id_list[:6]
                for enum_face_id in enum_face_id_list:
                    faces.append(face_boundary[i, j, k, enum_face_id.value])
                # 将每个单元面的值存储在列表中，其中包括物理边界条件、温度边界条件、温度值、热流值
                face_physics_boundary:List[PhysicsBoundaryCondition]= [] # 面的物理边界类型
                face_temperature_boundary:List[TemperatureBoundaryCondition] = [] # 面的温度边界条件类型
                face_temperature:List[Fp] = [] # 面的温度值
                face_flux:List[Fp] = [] # 面的热流值
                for face in faces:
                    face_physics_boundary.append(physics_boundary_condition[face.value].type)
                    face_temperature_boundary.append(temperature_boundary_condition[face.value].type)
                    face_temperature.append(temperature_boundary_condition[face.value].t)
                    face_flux.append(temperature_boundary_condition[face.value].heat_flux)
                
                # 根据边界条件修改（在原系数基础上修改）每个单元不同面的系数
                all_coefficient_list = [
                    [face_physics_boundary[0], face_temperature_boundary[0], mesh_coefficient.AE_ID.value,
                     Fp(face_temperature[0]), Fp(face_temperature[0]), x[i + 1], x_cell_centroid[i]],
                    [face_physics_boundary[1], face_temperature_boundary[1], mesh_coefficient.AW_ID.value,
                     Fp(face_temperature[1]), Fp(face_temperature[1]), x[i], x_cell_centroid[i - 1]],
                    [face_physics_boundary[2], face_temperature_boundary[2], mesh_coefficient.AS_ID.value,
                     Fp(face_temperature[2]), Fp(face_temperature[2]), y[j + 1], y_cell_centroid[j]],
                    [face_physics_boundary[3], face_temperature_boundary[3], mesh_coefficient.AN_ID.value,
                     Fp(face_temperature[3]), Fp(face_temperature[3]), y[j], y_cell_centroid[j - 1]]]
                if dim == 3:
                    all_coefficient_list.append(
                        [face_physics_boundary[4], face_temperature_boundary[4], mesh_coefficient.AT_ID.value,
                         Fp(face_temperature[4]), Fp(face_temperature[4]), z[k + 1], z_cell_centroid[k]])
                    all_coefficient_list.append(
                        [face_physics_boundary[5], face_temperature_boundary[5], mesh_coefficient.AB_ID.value,
                         Fp(face_temperature[5]), Fp(face_temperature[5]), z[k], z_cell_centroid[k - 1]])
               
                for value in all_coefficient_list:
                    if value[0] == PhysicsBoundaryID.INLET:
                        t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] -= t_coefficient[i, j, k, value[2]]
                        t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] -= Fp(2.0) * t_coefficient[
                            i, j, k, value[2]] * value[3]
                        t_coefficient[i, j, k, value[2]] = Fp(0.0)
                    elif value[0] == PhysicsBoundaryID.OUTLET:
                        t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] += t_coefficient[i, j, k, value[2]]
                        t_coefficient[i, j, k, value[2]] = Fp(0.0)
                    elif value[0] == PhysicsBoundaryID.WALL:
                        if value[1] == TemperatureBoundaryID.CONSTANT:
                            t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] -= t_coefficient[i, j, k, value[2]]
                            t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] -= Fp(2.0) * t_coefficient[\
                                i, j, k, value[2]] * value[3]
                            t_coefficient[i, j, k, value[2]] = Fp(0.0)
                        elif value[1] == TemperatureBoundaryID.HEAT_FLUX:
                            t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] += t_coefficient[i, j, k, value[2]]
                            t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] += t_coefficient[i, j, k, value[2]] * \
                                                                                    value[4] * Fp(2.0) * (
                                                                                                value[5] - value[
                                                                                            6]) / conductivity_coefficient
                            t_coefficient[i, j, k, value[2]] = Fp(0.0)
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                if i == 0:
                    t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] = Fp(1.0)
                    t_coefficient[i, j, k, mesh_coefficient.AE_ID.value] = Fp(0.0)
                    t_coefficient[i, j, k, mesh_coefficient.AW_ID.value] = Fp(0.0)
                    t_coefficient[i, j, k, mesh_coefficient.AN_ID.value] = Fp(0.0)
                    t_coefficient[i, j, k, mesh_coefficient.AS_ID.value] = Fp(0.0)
                    t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] = face_temperature[1]
                elif i == n_x_cell - 1:
                    t_coefficient[i, j, k, mesh_coefficient.AP_ID.value] = Fp(1.0)
                    t_coefficient[i, j, k, mesh_coefficient.AE_ID.value] = Fp(0.0)
                    t_coefficient[i, j, k, mesh_coefficient.AW_ID.value] = Fp(0.0)
                    t_coefficient[i, j, k, mesh_coefficient.AN_ID.value] = Fp(0.0)
                    t_coefficient[i, j, k, mesh_coefficient.AS_ID.value] = Fp(0.0)
                    t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] = face_temperature[0]
    return

def solve_velocity_boundary(dim,uf,vf,wf,u,v,w,face_id,face_boundary,\
                            n_x_cell,n_y_cell,n_z_cell,physics_boundary_condition,\
                            velocity_boundary_condition):
    '''考虑边界条件，求解面的速度值'''
    mf_in = Fp(0.0)
    mf_out = Fp(0.0)

    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                # 每个单元的faces面的边界条件
                faces = []
                enum_face_id_list = list(face_id)
                if dim == 2:
                    enum_face_id_list = enum_face_id_list[:4]
                elif dim == 3:
                    enum_face_id_list = enum_face_id_list[:6]
                for enum_face_id in enum_face_id_list:
                    faces.append(face_boundary[i, j, k, enum_face_id.value])

                # 定义列表用于存储单元面的速度边界条件、各个分量的速度值、各个分量的速度通量
                faces_physics_boundary:List[PhysicsBoundaryCondition] = []
                faces_u:List[Fp] = []
                faces_v:List[Fp] = []
                faces_w:List[Fp] = []
                faces_u_flux:List[Fp] = []
                faces_v_flux:List[Fp] = []
                faces_w_flux:List[Fp] = []
                for face in faces:
                    faces_physics_boundary.append(physics_boundary_condition[face.value].type)
                    faces_u.append(velocity_boundary_condition[face.value].u)
                    faces_v.append(velocity_boundary_condition[face.value].v)
                    faces_w.append(velocity_boundary_condition[face.value].w)
                    faces_u_flux.append(velocity_boundary_condition[face.value].u_flux)
                    faces_v_flux.append(velocity_boundary_condition[face.value].v_flux)
                    faces_w_flux.append(velocity_boundary_condition[face.value].w_flux)

                # 根据边界条件修改（在原系数基础上修改）每个单元不同面的系数
                all_coefficient_list = [
                    [faces_physics_boundary[0],faces_u[0],uf[i+1,j,k],u[i,j,k]],
                    [faces_physics_boundary[1],faces_u[1],uf[i,j,k],u[i,j,k]],
                    [faces_physics_boundary[2],faces_v[2],vf[i,j+1,k],v[i,j,k]],
                    [faces_physics_boundary[3],faces_v[3],vf[i,j,k],v[i,j,k]]]
                if dim == 3:
                    all_coefficient_list.append(
                        [faces_physics_boundary[4],faces_w[4],wf[i,j,k+1],w[i,j,k]])
                    all_coefficient_list.append(
                        [faces_physics_boundary[5],faces_w[5],wf[i,j,k],w[i,j,k]])

                for value in all_coefficient_list:
                    if value[0] == PhysicsBoundaryID.INLET:
                        value[2] = value[1]
                        mf_in += value[2]
                    elif value[0] == PhysicsBoundaryID.OUTLET:
                        value[2] = value[3]
                        mf_out += value[2]
                    elif value[0] == PhysicsBoundaryID.WALL:
                        value[2] = value[1]
    # QUESTION:不知道什么含义
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            mf_in += uf[0,j,k]
            mf_out += uf[n_x_cell,j,k]

def solve_momentum_coefficient(dim:int,dx,dy,dz,dt,conduction_scheme,\
                               mesh_coefficient,u_coefficient,v_coefficient,\
                               w_coefficient,n_x_cell,n_y_cell,n_z_cell,\
                               density,conductivity_coefficient,uf,vf,wf,u,v,w):
    '''求解动量方程系数'''
    idt = Fp(1.0)/dt
    area_x,area_y,area_z,vol = calculate_area_volume(dx, dy, dz)

    # 4、循环每个单元求解单元系数
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):    
                a_e = Fp(0.0)
                a_w = Fp(0.0)
                a_n = Fp(0.0)
                a_s = Fp(0.0)
                a_t = Fp(0.0)
                a_b = Fp(0.0)
                a_p = Fp(0.0)
                s_c = Fp(0.0)
                s_p = Fp(0.0)
                b_src = Fp(0.0)
                a_p0 = Fp(0.0)

                # 东边界
                if(i == n_x_cell-1):
                    rho = density[i,j,k]
                    mul = conductivity_coefficient[i,j,k]
                    mur = conductivity_coefficient[i,j,k]
                else:
                    rho = Fp(0.5)*(density[i,j,k]+density[i+1,j,k])
                    mul = conductivity_coefficient[i,j,k]
                    mur = conductivity_coefficient[i+1,j,k]
                ul = uf[i+1,j,k]
                ur = uf[i+1,j,k]
                a_e = calculate_face_coefficient(conduction_scheme, area_x, dx, ul, ur, mul, mur, rho, -Fp(1.0))

                # 西边界
                if(i == 0):
                    rho = density[i,j,k]
                    mul = conductivity_coefficient[i,j,k]
                    mur = conductivity_coefficient[i,j,k]
                else:
                    rho = Fp(0.5)*(density[i,j,k]+density[i-1,j,k])
                    mul = conductivity_coefficient[i-1,j,k]
                    mur = conductivity_coefficient[i,j,k]
                ul = uf[i-1,j,k]
                ur = uf[i,j,k]
                a_w = calculate_face_coefficient(conduction_scheme, area_x, dx, ul, ur, mul, mur, rho, Fp(1.0))

                # 北边界
                if(j == n_y_cell - 1):
                    rho = density[i,j,k]
                    mul = conductivity_coefficient[i,j,k]
                    mur = conductivity_coefficient[i,j,k]
                else:
                    rho = Fp(0.5)*(density[i,j,k]+density[i,j+1,k])
                    mul = conductivity_coefficient[i,j,k]
                    mur = conductivity_coefficient[i,j+1,k]
                vl = vf[i,j+1,k]
                vr = vf[i,j+1,k]
                a_n = calculate_face_coefficient(conduction_scheme, area_y, dy, vl, vr, mul, mur, rho, -Fp(1.0))

                # 南边界
                if(j == 0):
                    rho = density[i,j,k]
                    mul = conductivity_coefficient[i,j,k]
                    mur = conductivity_coefficient[i,j,k]
                else:
                    rho = Fp(0.5)*(density[i,j,k]+density[i,j-1,k])
                    mul = conductivity_coefficient[i,j-1,k]
                    mur = conductivity_coefficient[i,j,k]
                vl = vf[i,j,k]
                vr = vf[i,j,k]
                a_s = calculate_face_coefficient(conduction_scheme, area_y, dy, vl, vr, mul, mur, rho, Fp(1.0))

                if dim == 3:
                    # 上边界
                    if(k == n_z_cell - 1):
                        rho = density[i,j,k]
                        mul = conductivity_coefficient[i,j,k]
                        mur = conductivity_coefficient[i,j,k]
                    else:
                        rho = Fp(0.5)*(density[i,j,k]+density[i,j,k+1])
                        mul = conductivity_coefficient[i,j,k]
                        mur = conductivity_coefficient[i,j,k+1]
                    wl = wf[i,j,k+1]
                    wr = wf[i,j,k+1]
                    a_t = calculate_face_coefficient(conduction_scheme, area_z, dz, wl, wr, mul, mur, rho, -Fp(1.0))

                    # 下边界
                    if(k == 0):
                        rho = density[i,j,k]
                        mul = conductivity_coefficient[i,j,k]
                        mur = conductivity_coefficient[i,j,k]
                    else:
                        rho = Fp(0.5)*(density[i,j,k]+density[i,j,k-1])
                        mul = conductivity_coefficient[i,j,k-1]
                        mur = conductivity_coefficient[i,j,k]
                    wl = wf[i,j,k]
                    wr = wf[i,j,k]
                    a_b = calculate_face_coefficient(conduction_scheme, area_z, dz, wl, wr, mul, mur, rho, Fp(1.0))

                # 计算系数
                rho = density[i,j,k]
                a_p0 = rho*vol*idt
                a_p = a_p + a_e + a_w + a_n + a_s + a_p0 - s_p*vol
                if dim == 3:
                    a_p = a_p + a_t + a_b 
                u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] = a_p
                u_coefficient[i,j,k,mesh_coefficient.AE_ID.value] = -a_e
                u_coefficient[i,j,k,mesh_coefficient.AW_ID.value] = -a_w
                u_coefficient[i,j,k,mesh_coefficient.AN_ID.value] = -a_n
                u_coefficient[i,j,k,mesh_coefficient.AS_ID.value] = -a_s
                if dim==3:
                    u_coefficient[i,j,k,mesh_coefficient.AT_ID.value]  = -a_t
                    u_coefficient[i,j,k,mesh_coefficient.AB_ID.value]  = -a_b
                b_src  = s_c*vol + a_p0*u[i,j,k]
                u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] = b_src

                v_coefficient[i,j,k,mesh_coefficient.AP_ID.value] = a_p
                v_coefficient[i,j,k,mesh_coefficient.AE_ID.value] = -a_e
                v_coefficient[i,j,k,mesh_coefficient.AW_ID.value] = -a_w
                v_coefficient[i,j,k,mesh_coefficient.AN_ID.value] = -a_n
                v_coefficient[i,j,k,mesh_coefficient.AS_ID.value] = -a_s
                if dim==3:
                    v_coefficient[i,j,k,mesh_coefficient.AT_ID.value]  = -a_t
                    v_coefficient[i,j,k,mesh_coefficient.AB_ID.value]  = -a_b
                b_src  = s_c*vol + a_p0*v[i,j,k]
                v_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] = b_src

                if dim == 3:
                    w_coefficient[i,j,k,mesh_coefficient.AP_ID.value] = a_p
                    w_coefficient[i,j,k,mesh_coefficient.AE_ID.value] = -a_e
                    w_coefficient[i,j,k,mesh_coefficient.AW_ID.value] = -a_w
                    w_coefficient[i,j,k,mesh_coefficient.AN_ID.value] = -a_n
                    w_coefficient[i,j,k,mesh_coefficient.AS_ID.value] = -a_s
                    if dim==3:
                        w_coefficient[i,j,k,mesh_coefficient.AT_ID.value]  = -a_t
                        w_coefficient[i,j,k,mesh_coefficient.AB_ID.value]  = -a_b
                    b_src  = s_c*vol + a_p0*w[i,j,k]
                    w_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] = b_src

def modify_momentum_boundary_coefficient(dim,n_x_cell,n_y_cell,n_z_cell,\
                                         face_id,physics_boundary_condition,\
                                         velocity_boundary_condition,face_boundary,\
                                         u_coefficient,mesh_coefficient,direction:Direction):
    '''考虑边界条件，修改动量方程系数'''
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                bcid_e = face_boundary[i,j,k,face_id.EAST.value]
                bcid_w = face_boundary[i,j,k,face_id.WEST.value]
                bcid_n = face_boundary[i,j,k,face_id.NORTH.value]
                bcid_s = face_boundary[i,j,k,face_id.SOUTH.value]
                if dim ==3:
                    bcid_t = face_boundary[i,j,k,face_id.TOP.value]
                    bcid_b = face_boundary[i,j,k,face_id.BOTTOM.value]
                
                bc_e = physics_boundary_condition[bcid_e.value].type
                bc_w = physics_boundary_condition[bcid_w.value].type
                bc_n = physics_boundary_condition[bcid_n.value].type
                bc_s = physics_boundary_condition[bcid_s.value].type
                if dim ==3:
                    bc_t = physics_boundary_condition[bcid_t.value].type
                    bc_b = physics_boundary_condition[bcid_b.value].type

                if direction == Direction.X:
                    bc_ue = velocity_boundary_condition[bcid_e.value].u
                    bc_uw = velocity_boundary_condition[bcid_w.value].u
                    bc_un = velocity_boundary_condition[bcid_n.value].u
                    bc_us = velocity_boundary_condition[bcid_s.value].u
                    if dim == 3:
                        bc_ut = velocity_boundary_condition[bcid_t.value].u
                        bc_ub = velocity_boundary_condition[bcid_b.value].u
                if direction == Direction.Y:
                    bc_ue = velocity_boundary_condition[bcid_e.value].v
                    bc_uw = velocity_boundary_condition[bcid_w.value].v
                    bc_un = velocity_boundary_condition[bcid_n.value].v
                    bc_us = velocity_boundary_condition[bcid_s.value].v
                    if dim == 3:
                        bc_ut = velocity_boundary_condition[bcid_t.value].v
                        bc_ub = velocity_boundary_condition[bcid_b.value].v
                if direction == Direction.Z:
                    bc_ue = velocity_boundary_condition[bcid_e.value].w
                    bc_uw = velocity_boundary_condition[bcid_w.value].w
                    bc_un = velocity_boundary_condition[bcid_n.value].w
                    bc_us = velocity_boundary_condition[bcid_s.value].w
                    if dim == 3:
                        bc_ut = velocity_boundary_condition[bcid_t.value].w
                        bc_ub = velocity_boundary_condition[bcid_b.value].w
                
                # 东面
                if bc_e == PhysicsBoundaryID.INLET:
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AE_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AW_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AE_ID.value]/3.0
                    u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AE_ID.value]*bc_ue/3.0
                    u_coefficient[i,j,k,mesh_coefficient.AE_ID.value]   = 0.0
                elif bc_e == PhysicsBoundaryID.OUTLET:
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AE_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AE_ID.value] = 0.0
                elif bc_e == PhysicsBoundaryID.WALL:                    
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AE_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AW_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AE_ID.value]/3.0
                    u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AE_ID.value]*bc_ue/3.0
                    u_coefficient[i,j,k,mesh_coefficient.AE_ID.value] = 0.0
                # 西面
                if bc_w == PhysicsBoundaryID.INLET:
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AW_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AE_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AW_ID.value]/3.0
                    u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AW_ID.value]*bc_uw/3.0
                    u_coefficient[i,j,k,mesh_coefficient.AW_ID.value]   = 0.0
                elif bc_w == PhysicsBoundaryID.OUTLET:
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AW_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AW_ID.value] = 0.0
                elif bc_w == PhysicsBoundaryID.WALL:                    
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AW_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AE_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AW_ID.value]/3.0
                    u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AW_ID.value]*bc_uw/3.0
                    u_coefficient[i,j,k,mesh_coefficient.AW_ID.value] = 0.0
                # 北边
                if bc_n == PhysicsBoundaryID.INLET:
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AN_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AS_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AN_ID.value]/3.0
                    u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AN_ID.value]*bc_un/3.0
                    u_coefficient[i,j,k,mesh_coefficient.AN_ID.value]   = 0.0
                elif bc_n == PhysicsBoundaryID.OUTLET:
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AN_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AN_ID.value] = 0.0
                elif bc_n == PhysicsBoundaryID.WALL:                    
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AN_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AS_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AN_ID.value]/3.0
                    u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AN_ID.value]*bc_un/3.0
                    u_coefficient[i,j,k,mesh_coefficient.AN_ID.value] = 0.0
                # 南边
                if bc_s == PhysicsBoundaryID.INLET:
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AS_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AN_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AS_ID.value]/3.0
                    u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AS_ID.value]*bc_us/3.0
                    u_coefficient[i,j,k,mesh_coefficient.AS_ID.value]   = 0.0
                elif bc_s == PhysicsBoundaryID.OUTLET:
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AS_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AS_ID.value] = 0.0
                elif bc_s == PhysicsBoundaryID.WALL:                    
                    u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AS_ID.value]
                    u_coefficient[i,j,k,mesh_coefficient.AN_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AS_ID.value]/3.0
                    u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AS_ID.value]*bc_us/3.0
                    u_coefficient[i,j,k,mesh_coefficient.AS_ID.value] = 0.0
                if dim == 3:
                    # 上边
                    if bc_t == PhysicsBoundaryID.INLET:
                        u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AT_ID.value]
                        u_coefficient[i,j,k,mesh_coefficient.AB_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AT_ID.value]/3.0
                        u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AT_ID.value]*bc_ut/3.0
                        u_coefficient[i,j,k,mesh_coefficient.AT_ID.value]   = 0.0
                    elif bc_t == PhysicsBoundaryID.OUTLET:
                        u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AT_ID.value]
                        u_coefficient[i,j,k,mesh_coefficient.AT_ID.value] = 0.0
                    elif bc_t == PhysicsBoundaryID.WALL:                    
                        u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AT_ID.value]
                        u_coefficient[i,j,k,mesh_coefficient.AB_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AT_ID.value]/3.0
                        u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AT_ID.value]*bc_ut/3.0
                        u_coefficient[i,j,k,mesh_coefficient.AT_ID.value] = 0.0
                    # 下边
                    if bc_b == PhysicsBoundaryID.INLET:
                        u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AB_ID.value]
                        u_coefficient[i,j,k,mesh_coefficient.AT_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AB_ID.value]/3.0
                        u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AB_ID.value]*bc_ub/3.0
                        u_coefficient[i,j,k,mesh_coefficient.AB_ID.value]   = 0.0
                    elif bc_b == PhysicsBoundaryID.OUTLET:
                        u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AB_ID.value]
                        u_coefficient[i,j,k,mesh_coefficient.AB_ID.value] = 0.0
                    elif bc_b == PhysicsBoundaryID.WALL:                    
                        u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] -= 2.0*u_coefficient[i,j,k,mesh_coefficient.AB_ID.value]
                        u_coefficient[i,j,k,mesh_coefficient.AT_ID.value] += u_coefficient[i,j,k,mesh_coefficient.AB_ID.value]/3.0
                        u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] -= 8.0*u_coefficient[i,j,k,mesh_coefficient.AB_ID.value]*bc_ub/3.0
                        u_coefficient[i,j,k,mesh_coefficient.AB_ID.value] = 0.0        

def solve_momentum_gradp(dim,p_outlet,p,face_boundary,face_id,physics_boundary_condition,\
                         mesh_coefficient,n_x_cell,n_y_cell,n_z_cell,\
                         dx,dy,dz,u_coefficient,v_coefficient,w_coefficient):
    '''求解动量方程压力梯度'''
    area_x, area_y, area_z, vol = calculate_area_volume(dx, dy, dz)

    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                bcid_e = face_boundary[i,j,k,face_id.EAST.value]
                bcid_w = face_boundary[i,j,k,face_id.WEST.value]
                bcid_n = face_boundary[i,j,k,face_id.NORTH.value]
                bcid_s = face_boundary[i,j,k,face_id.SOUTH.value]
                if dim ==3:
                    bcid_t = face_boundary[i,j,k,face_id.TOP.value]
                    bcid_b = face_boundary[i,j,k,face_id.BOTTOM.value]
                
                bc_e = physics_boundary_condition[bcid_e.value].type
                bc_w = physics_boundary_condition[bcid_w.value].type
                bc_n = physics_boundary_condition[bcid_n.value].type
                bc_s = physics_boundary_condition[bcid_s.value].type
                if dim ==3:
                    bc_t = physics_boundary_condition[bcid_t.value].type
                    bc_b = physics_boundary_condition[bcid_b.value].type

                # x方向
                if bc_e == PhysicsBoundaryID.NONE and bc_w == PhysicsBoundaryID.NONE:
                    # 内部单元
                    pl = p[i,j,k]
                    pr = p[i+1,j,k]
                else:
                    # 东(右)侧边界
                    if bc_e == PhysicsBoundaryID.INLET or bc_e == PhysicsBoundaryID.WALL:
                        pl = p[i-1,j,k]
                        pr = p[i,j,k]
                    elif bc_e == PhysicsBoundaryID.OUTLET:
                        pl = p[i-1,j,k]
                        pr = p_outlet
                    # 西(左)侧边界
                    if bc_w == PhysicsBoundaryID.INLET or bc_w == PhysicsBoundaryID.WALL:
                        pl = p[i,j,k]
                        pr = p[i+1,j,k]
                    elif bc_w == PhysicsBoundaryID.OUTLET:
                        pl = p_outlet
                        pr = p[i+1,j,k]
                u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] = u_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] + 0.5 * (pl - pr) * area_x

                # y方向
                if bc_n == PhysicsBoundaryID.NONE and bc_s == PhysicsBoundaryID.NONE:
                    # 内部单元
                    pl = p[i,j-1,k]
                    pr = p[i,j+1,k]
                else:
                    # 北(上)侧边界
                    if bc_n == PhysicsBoundaryID.INLET or bc_n == PhysicsBoundaryID.WALL:
                        pl = p[i,j-1,k]
                        pr = p[i,j,k]
                    elif bc_n == PhysicsBoundaryID.OUTLET:
                        pl = p[i,j-1,k]
                        pr = p_outlet
                    # 南(下)侧边界
                    if bc_s == PhysicsBoundaryID.INLET or bc_s == PhysicsBoundaryID.WALL:
                        pl = p[i,j,k]
                        pr = p[i,j+1,k]
                    elif bc_s == PhysicsBoundaryID.OUTLET:
                        pl = p_outlet
                        pr = p[i,j+1,k]
                v_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] = v_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] + 0.5 * (pl - pr) * area_y

                if dim == 3:
                    # z方向
                    if bc_t == PhysicsBoundaryID.NONE and bc_b == PhysicsBoundaryID.NONE:
                        # 内部单元
                        pl = p[i,j,k-1]
                        pr = p[i,j,k+1]
                    else:
                        # 上侧边界
                        if bc_t == PhysicsBoundaryID.INLET or bc_t == PhysicsBoundaryID.WALL:
                            pl = p[i,j,k-1]
                            pr = p[i,j,k]
                        elif bc_t == PhysicsBoundaryID.OUTLET:
                            pl = p[i,j,k-1]
                            pr = p_outlet
                        # 下侧边界
                        if bc_b == PhysicsBoundaryID.INLET or bc_b == PhysicsBoundaryID.WALL:
                            pl = p[i,j,k]
                            pr = p[i,j,k+1]
                        elif bc_b == PhysicsBoundaryID.OUTLET:
                            pl = p_outlet
                            pr = p[i,j,k+1]
                    w_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] = w_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] + 0.5 * (pl - pr) * area_z

def rhie_chow_face_velocity(dim,old_u,old_v,old_w,n_x_cell,n_y_cell,n_z_cell,dx,dy,dz,u,v,w,uf,vf,wf,\
                            p,u_coefficient,v_coefficient,w_coefficient,density,dt,\
                            face_boundary,face_id,mesh_coefficient,p_outlet,physics_boundary_condition):
    '''使用rhie_chow求解面速度'''
    area_x, area_y, area_z, vol = calculate_area_volume(dx, dy, dz)
    idt = 1.0 / dt
    old_u = np.copy(u)
    old_v = np.copy(v)
    old_w = np.copy(w)

    # x面速度
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(1,n_x_cell):
                bcid_e = face_boundary[i,j,k,face_id.EAST.value]
                bcid_w = face_boundary[i,j,k,face_id.WEST.value]
                bc_e = physics_boundary_condition[bcid_e.value].type
                bc_w = physics_boundary_condition[bcid_w.value].type

                uf_old = uf[i,j,k]
                pe = p[i,j,k]
                pw = p[i-1,j,k]

                # Set the pww and pee
                if i==1:
                    pww = Fp(0.0)
                else:
                    pww = p[i-2,j,k]

                if i==n_x_cell-1:
                    pee = Fp(0.0)
                else:
                    pee = p[i+1,j,k]

                # 西边
                if bc_w == PhysicsBoundaryID.INLET or bc_w == PhysicsBoundaryID.WALL:
                    pww = p[i-1,j,k]
                elif bc_w == PhysicsBoundaryID.OUTLET:
                    pww = p_outlet
                # 东边
                if bc_e == PhysicsBoundaryID.INLET or bc_e == PhysicsBoundaryID.WALL:
                    pee = p[i,j,k]
                elif bc_e == PhysicsBoundaryID.OUTLET:
                    pee = p_outlet
                
                # QUESTION:不知道什么含义
                dpm = (pe-pww) * 0.5
                dpp = (pee-pw) * 0.5
                dpf = (pe-pw)

                uf[i,j,k] = ( \
                    (u_coefficient[i-1,j,k,mesh_coefficient.AP_ID.value] * u[i-1,j,k] + u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] * u[i,j,k]) \
                    / (u_coefficient[i-1,j,k,mesh_coefficient.AP_ID.value] + u_coefficient[i,j,k,mesh_coefficient.AP_ID.value]) \
                    + ( \
                          0.5 * dpm / u_coefficient[i-1,j,k,mesh_coefficient.AP_ID.value] \
                        + 0.5 * dpp / u_coefficient[i,j,k,mesh_coefficient.AP_ID.value] \
                        - 0.5 * dpf * (1.0 / u_coefficient[i-1,j,k,mesh_coefficient.AP_ID.value] + 1.0 / u_coefficient[i,j,k,mesh_coefficient.AP_ID.value]) \
                      ) * area_x \
                )
                # Transient term
                cf0 = density[i,j,k] * idt
                cf = density[i,j,k] * idt
                df = 0.5 * (vol / u_coefficient[i-1,j,k,mesh_coefficient.AP_ID.value] + vol / u_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                df_hat = df / (1.0 + cf * df)
                uf[i,j,k] = uf[i,j,k] + cf0 * df_hat * (uf_old - 0.5 * (old_u[i-1,j,k] + old_u[i,j,k]))

    # y面速度
    for k in range(n_z_cell):
        for j in range(1,n_y_cell):
            for i in range(n_x_cell):
                bcid_n = face_boundary[i,j,k,face_id.NORTH.value]
                bcid_s = face_boundary[i,j,k,face_id.SOUTH.value]
                bc_n = physics_boundary_condition[bcid_n.value].type
                bc_s = physics_boundary_condition[bcid_s.value].type

                vf_old = vf[i,j,k]
                pe = p[i,j,k]
                pw = p[i,j-1,k]

                # Set the pww and pee
                if j==1:
                    pww = Fp(0.0)
                else:
                    pww = p[i,j-2,k]

                if j==n_y_cell-1:
                    pee = Fp(0.0)
                else:
                    pee = p[i,j+1,k]

                # 西边
                if bc_s == PhysicsBoundaryID.INLET or bc_s == PhysicsBoundaryID.WALL:
                    pww = p[i,j-1,k]
                elif bc_s == PhysicsBoundaryID.OUTLET:
                    pww = p_outlet
                # 东边
                if bc_n == PhysicsBoundaryID.INLET or bc_n == PhysicsBoundaryID.WALL:
                    pee = p[i,j,k]
                elif bc_n == PhysicsBoundaryID.OUTLET:
                    pee = p_outlet
                
                # QUESTION:不知道什么含义
                dpm = (pe-pww) * 0.5
                dpp = (pee-pw) * 0.5
                dpf = (pe-pw)

                vf[i,j,k] = ( \
                    (v_coefficient[i,j-1,k,mesh_coefficient.AP_ID.value] * v[i,j-1,k] + v_coefficient[i,j,k,mesh_coefficient.AP_ID.value] * u[i,j,k]) \
                    / (v_coefficient[i,j-1,k,mesh_coefficient.AP_ID.value] + v_coefficient[i,j,k,mesh_coefficient.AP_ID.value]) \
                    + ( \
                          0.5 * dpm / v_coefficient[i,j-1,k,mesh_coefficient.AP_ID.value] \
                        + 0.5 * dpp / v_coefficient[i,j,k,mesh_coefficient.AP_ID.value] \
                        - 0.5 * dpf * (1.0 / v_coefficient[i,j-1,k,mesh_coefficient.AP_ID.value] + 1.0 / v_coefficient[i,j,k,mesh_coefficient.AP_ID.value]) \
                      ) * area_y \
                )
                # QUESTION:瞬态项
                cf0 = density[i,j,k] * idt
                cf = density[i,j,k] * idt
                df = 0.5 * (vol / v_coefficient[i,j-1,k,mesh_coefficient.AP_ID.value] + vol / v_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                df_hat = df / (1.0 + cf * df)
                vf[i,j,k] = vf[i,j,k] + cf0 * df_hat * (vf_old - 0.5 * (old_v[i,j-1,k] + old_v[i,j,k]))

    if dim == 3:
        # z面速度
        for k in range(1,n_z_cell):
            for j in range(n_y_cell):
                for i in range(n_x_cell):
                    bcid_t = face_boundary[i,j,k,face_id.TOP.value]
                    bcid_b = face_boundary[i,j,k,face_id.BOTTOM.value]
                    bc_t = physics_boundary_condition[bcid_t.value].type
                    bc_b = physics_boundary_condition[bcid_b.value].type

                    wf_old = wf[i,j,k]
                    pe = p[i,j,k]
                    pw = p[i,j,k-1]

                    # Set the pww and pee
                    if k==1:
                        pww = Fp(0.0)
                    else:
                        pww = p[i,j,k-2]

                    if k==n_z_cell-1:
                        pee = Fp(0.0)
                    else:
                        pee = p[i,j,k+1]

                    # 上边
                    if bc_t == PhysicsBoundaryID.INLET or bc_t == PhysicsBoundaryID.WALL:
                        pww = p[i,j,k-1]
                    elif bc_t == PhysicsBoundaryID.OUTLET:
                        pww = p_outlet
                    # 下边
                    if bc_b == PhysicsBoundaryID.INLET or bc_b == PhysicsBoundaryID.WALL:
                        pee = p[i,j,k]
                    elif bc_b == PhysicsBoundaryID.OUTLET:
                        pee = p_outlet
                    
                    # QUESTION:不知道什么含义
                    dpm = (pe-pww) * 0.5
                    dpp = (pee-pw) * 0.5
                    dpf = (pe-pw)

                    wf[i,j,k] = ( \
                        (w_coefficient[i,j,k-1,mesh_coefficient.AP_ID.value] * w[i,j,k-1] + w_coefficient[i,j,k,mesh_coefficient.AP_ID.value] * u[i,j,k]) \
                        / (w_coefficient[i,j,k-1,mesh_coefficient.AP_ID.value] + w_coefficient[i,j,k,mesh_coefficient.AP_ID.value]) \
                        + ( \
                            0.5 * dpm / w_coefficient[i,j,k-1,mesh_coefficient.AP_ID.value] \
                            + 0.5 * dpp / w_coefficient[i,j,k,mesh_coefficient.AP_ID.value] \
                            - 0.5 * dpf * (1.0 / w_coefficient[i,j,k-1,mesh_coefficient.AP_ID.value] + 1.0 / w_coefficient[i,j,k,mesh_coefficient.AP_ID.value]) \
                        ) * area_z \
                    )
                    # QUESTION:瞬态项
                    cf0 = density[i,j,k] * idt
                    cf = density[i,j,k] * idt
                    df = 0.5 * (vol / w_coefficient[i,j,k-1,mesh_coefficient.AP_ID.value] + vol / w_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                    df_hat = df / (1.0 + cf * df)
                    wf[i,j,k] = wf[i,j,k] + cf0 * df_hat * (wf_old - 0.5 * (old_w[i,j,k-1] + old_w[i,j,k]))

def solve_pressure_coefficient(dim,n_x_cell,n_y_cell,n_z_cell,dx,dy,dz,\
                               physics_boundary_condition,face_boundary,face_id,\
                               u_coefficient,v_coefficient,w_coefficient,density,\
                               mesh_coefficient,uf,vf,wf,p_coefficient):
    '''求解压力系数'''
    area_x, area_y, area_z, vol = calculate_area_volume(dx, dy, dz)

    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                # enum_face_id_list = list(face_id)
                # if dim == 2:
                #     enum_face_id_list = enum_face_id_list[:4]
                # elif dim == 3:
                #     enum_face_id_list = enum_face_id_list[:6]

                # bcid_e = face_boundary[i,j,k,enum_face_id_list[0].value]
                # bcid_w = face_boundary[i,j,k,enum_face_id_list[1].value]
                # bcid_n = face_boundary[i,j,k,enum_face_id_list[2].value]
                # bcid_s = face_boundary[i,j,k,enum_face_id_list[3].value]
                # if dim == 3:
                #     bcid_t = face_boundary[i,j,k,enum_face_id_list[4].value]
                #     bcid_b = face_boundary[i,j,k,enum_face_id_list[5].value]
                bcid_e = face_boundary[i,j,k,face_id.EAST.value]
                bcid_w = face_boundary[i,j,k,face_id.WEST.value]
                bcid_n = face_boundary[i,j,k,face_id.NORTH.value]
                bcid_s = face_boundary[i,j,k,face_id.SOUTH.value]
                if dim == 3:
                    bcid_t = face_boundary[i,j,k,face_id.TOP.value]
                    bcid_b = face_boundary[i,j,k,face_id.BOTTOM.value]

                bc_e = physics_boundary_condition[bcid_e.value].type
                bc_w = physics_boundary_condition[bcid_w.value].type
                bc_n = physics_boundary_condition[bcid_n.value].type
                bc_s = physics_boundary_condition[bcid_s.value].type
                if dim == 3:
                    bc_t = physics_boundary_condition[bcid_t.value].type
                    bc_b = physics_boundary_condition[bcid_b.value].type

                a_e = Fp(0.0)
                a_w = Fp(0.0)
                a_n = Fp(0.0)
                a_s = Fp(0.0)
                a_p = Fp(0.0)
                b_src = Fp(0.0)
                if dim == 3:
                    a_t = Fp(0.0)
                    a_b = Fp(0.0)
                
                rho_w = Fp(0.0)
                rho_e = Fp(0.0)
                rho_s = Fp(0.0)
                rho_n = Fp(0.0)
                if dim == 3:
                    rho_t = Fp(0.0)
                    rho_b = Fp(0.0)
                
                # 西边
                if bc_w != PhysicsBoundaryID.NONE or i == 0:
                    a_w = 0.0
                    rho_w = density[i,j,k]
                else:
                    rho_w = 0.5 * (density[i,j,k] + density[i-1,j,k])
                    a_w = rho_w * area_x * area_x * 0.5 * (1.0 / u_coefficient[i-1,j,k,mesh_coefficient.AP_ID.value] + 1.0 / u_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                
                # 东边
                if bc_e != PhysicsBoundaryID.NONE or i == n_x_cell - 1:
                    a_e = 0.0
                    rho_e = density[i,j,k]
                else:
                    rho_e = 0.5 * (density[i,j,k] + density[i+1,j,k])
                    a_e = rho_e * area_x * area_x * 0.5 * (1.0 / u_coefficient[i+1,j,k,mesh_coefficient.AP_ID.value] + 1.0 / u_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                
                # 南边
                if bc_s != PhysicsBoundaryID.NONE or j == 0:
                    a_s = 0.0
                    rho_s = density[i,j,k]
                else:
                    rho_s = 0.5 * (density[i,j,k] + density[i,j-1,k])
                    a_s = rho_s * area_y * area_y * 0.5 * (1.0 / v_coefficient[i,j-1,k,mesh_coefficient.AP_ID.value] + 1.0 / v_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                
                # 北边
                if bc_n != PhysicsBoundaryID.NONE or j == n_y_cell - 1:
                    a_n = 0.0
                    rho_n = density[i,j,k]
                else:
                    rho_n = 0.5 * (density[i,j,k] + density[i,j+1,k])
                    a_n = rho_n * area_y * area_y * 0.5 * (1.0 / v_coefficient[i,j+1,k,mesh_coefficient.AP_ID.value] + 1.0 / v_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                
                if dim == 3:
                    # 上边
                    if bc_t != PhysicsBoundaryID.NONE or k == 0:
                        a_t = 0.0
                        rho_t = density[i,j,k]
                    else:
                        rho_t = 0.5 * (density[i,j,k] + density[i,j,k-1])
                        a_s = rho_s * area_z * area_z * 0.5 * (1.0 / w_coefficient[i,j,k-1,mesh_coefficient.AP_ID.value] + 1.0 / w_coefficient[i,j,k,mesh_coefficient.AP_ID.value])

                    # 下边
                    if bc_b != PhysicsBoundaryID.NONE or k == n_z_cell - 1:
                        a_b = 0.0
                        rho_b = density[i,j,k]
                    else:
                        rho_b = 0.5 * (density[i,j,k] + density[i,j,k+1])
                        a_b = rho_b * area_z * area_z * 0.5 * (1.0 / w_coefficient[i,j,k+1,mesh_coefficient.AP_ID.value] + 1.0 / w_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                
                a_p = a_e + a_w + a_n + a_s
                if dim == 3:
                    a_p += a_t + a_b

                b_src = (rho_w * uf[i,j,k] - rho_e * uf[i+1,j,k]) * area_x + \
                       (rho_s * vf[i,j,k] - rho_n * vf[i,j+1,k]) * area_y
                if dim == 3:
                    b_src += (rho_t * wf[i,j,k] - rho_b * wf[i,j,k+1]) * area_z

                p_coefficient[i,j,k,mesh_coefficient.AP_ID.value] = a_p
                p_coefficient[i,j,k,mesh_coefficient.AE_ID.value] = -a_e
                p_coefficient[i,j,k,mesh_coefficient.AW_ID.value] = -a_e
                p_coefficient[i,j,k,mesh_coefficient.AN_ID.value] = -a_n
                p_coefficient[i,j,k,mesh_coefficient.AS_ID.value] = -a_s
                if dim == 3:
                    p_coefficient[i,j,k,mesh_coefficient.AT_ID.value] = -a_t
                    p_coefficient[i,j,k,mesh_coefficient.AB_ID.value] = -a_b
                p_coefficient[i,j,k,mesh_coefficient.ABRC_ID.value] = b_src

def correct_pressure(n_x_cell:int, n_y_cell:int, n_z_cell:int, relax_p, pp, p):
    '''修正压力'''
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                p[i,j,k] += relax_p * pp[i, j, k]

def correct_velocity(dim,pp_outlet,dx,dy,dz,n_x_cell,n_y_cell,n_z_cell,\
                     u,v,w,uf,vf,wf,u_coefficient,v_coefficient,w_coefficient,\
                     mesh_coefficient,face_id,face_boundary,physics_boundary_condition,\
                        pp):
    '''修正速度'''
    area_x,area_y,area_z,vol = calculate_area_volume(dx,dy,dz)
    
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                bcid_e = face_boundary[i,j,k,face_id.EAST.value]
                bcid_w = face_boundary[i,j,k,face_id.WEST.value]

                bc_e = physics_boundary_condition[bcid_e.value].type
                bc_w = physics_boundary_condition[bcid_w.value].type

                if i == 0:
                    pl = Fp(0.0)
                else:
                    pl = pp[i-1,j,k]

                if i == n_x_cell - 1:
                    pr = Fp(0.0)
                else:
                    pr = pp[i+1,j,k]

                if bc_w == PhysicsBoundaryID.INLET or bc_w == PhysicsBoundaryID.WALL:
                    pl = pp[i,j,k]
                    pr = pp[i+1,j,k]
                elif bc_w == PhysicsBoundaryID.OUTLET:
                    pl = pp_outlet
                    pr = pp[i+1,j,k]
                
                if bc_e == PhysicsBoundaryID.INLET or bc_e == PhysicsBoundaryID.WALL:
                    pl = pp[i-1,j,k]
                    pr = pp[i,j,k]
                elif bc_e == PhysicsBoundaryID.OUTLET:
                    pl = pp[i-1,j,k]
                    pr = pp_outlet

                d = area_x * 0.5 / u_coefficient[i,j,k,mesh_coefficient.AP_ID.value]
                u[i,j,k] += d * (pl - pr)
    # y轴面
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                bcid_s = face_boundary[i,j,k,face_id.SOUTH.value]
                bcid_n = face_boundary[i,j,k,face_id.NORTH.value]

                bc_s = physics_boundary_condition[bcid_s.value].type
                bc_n = physics_boundary_condition[bcid_n.value].type

                if j == 0:
                    pl = Fp(0.0)
                else:
                    pl = pp[i,j-1,k]

                if j == n_y_cell - 1:
                    pr = Fp(0.0)
                else:
                    pr = pp[i,j+1,k]

                if bc_n == PhysicsBoundaryID.INLET or bc_n == PhysicsBoundaryID.WALL:
                    pl = pp[i,j-1,k]
                    pr = pp[i,j,k]
                elif bc_n == PhysicsBoundaryID.OUTLET:
                    pl = pp[i,j-1,k]
                    pr = pp_outlet
                
                if bc_s == PhysicsBoundaryID.INLET or bc_s == PhysicsBoundaryID.WALL:
                    pl = pp[i,j,k]
                    pr = pp[i,j+1,k]
                elif bc_s == PhysicsBoundaryID.OUTLET:
                    pl = pp_outlet
                    pr = pp[i,j+1,k]

                d = area_y * 0.5 / v_coefficient[i,j,k,mesh_coefficient.AP_ID.value]
                v[i,j,k] += d * (pl - pr)
    
    if dim == 3:
        for k in range(n_z_cell):
            for j in range(n_y_cell):
                for i in range(n_x_cell):
                    bcid_t = face_boundary[i, j, k, face_id.TOP.value]
                    bcid_b = face_boundary[i, j, k, face_id.BOTTOM.value]
                    
                    bc_t = physics_boundary_condition[bcid_t.value].type
                    bc_b = physics_boundary_condition[bcid_b.value].type

                    if k == 0:
                        pl = Fp(0.0)
                    else:
                        pl = pp[i,j,k-1]

                    if k == n_z_cell - 1:
                        pr = Fp(0.0)
                    else:
                        pr = pp[i,j,k+1]

                    if bc_t == PhysicsBoundaryID.INLET or bc_t == PhysicsBoundaryID.WALL:
                        pl = pp[i,j,k-1]
                        pr = pp[i,j,k]
                    elif bc_t == PhysicsBoundaryID.OUTLET:
                        pl = pp[i,j,k-1]
                        pr = pp_outlet
                    
                    if bc_b == PhysicsBoundaryID.INLET or bc_b == PhysicsBoundaryID.WALL:
                        pl = pp[i,j,k]
                        pr = pp[i,j,k+1]
                    elif bc_b == PhysicsBoundaryID.OUTLET:
                        pl = pp_outlet
                        pr = pp[i,j,k+1]

                    d = area_z * 0.5 / w_coefficient[i,j,k,mesh_coefficient.AP_ID.value]
                    w[i,j,k] += d * (pl - pr)
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                d = area_x * 0.5 * (1.0 / u_coefficient[i-1,j,k,mesh_coefficient.AP_ID.value] + 1.0 / u_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                uf[i,j,k] += d * (pp[i-1,j,k] - pp[i,j,k])
    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                d = area_y * 0.5 * (1.0 / v_coefficient[i,j-1,k,mesh_coefficient.AP_ID.value] + 1.0 / u_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                vf[i,j,k] += d * (pp[i,j-1,k] - pp[i,j,k])

    if dim == 3:
        for k in range(n_z_cell):
            for j in range(n_y_cell):
                for i in range(n_x_cell):
                    d = area_z * 0.5 * (1.0 / w_coefficient[i,j,k-1,mesh_coefficient.AP_ID.value] + 1.0 / u_coefficient[i,j,k,mesh_coefficient.AP_ID.value])
                    wf[i,j,k] += d * (pp[i,j,k-1] - pp[i,j,k])
