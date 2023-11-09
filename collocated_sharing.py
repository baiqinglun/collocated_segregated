import numpy as np
from fp import Fp
from mesh import MeshManager
from tool import calculate_area_volume, calculate_face_coefficient
from boundary import FluidBoundaryCondition, PhysicsBoundaryCondition, TemperatureBoundaryCondition
from typing import List
from case import CaseManager, MeshCoefficient2D, MeshCoefficient3D
from boundary_id import PhysicsBoundaryID,TemperatureBoundaryID,BoundaryLimitID

# 分离求解器中函数的实现细节
def solve_conduction_coefficient(dx, dy, dz, t_coefficient, mesh_coefficient, dim,
                                 n_x_cell, n_y_cell, n_z_cell,
                                 dt, specific_heat_capacity, conductivity_coefficient, source_term,
                                 density, t, u, v, w):
    # 获取Cell的面积和体积
    area_x, area_y, area_z, vol = calculate_area_volume(dx, dy, dz)

    for k in range(n_z_cell):
        for j in range(n_y_cell):
            for i in range(n_x_cell):
                # 初始化系数
                a_E = a_W = a_S = a_N = a_T = a_B = a_P = Fp(0.0)
                s_C = source_term
                s_P = Fp(0.0)
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
                a_E = calculate_face_coefficient(area_x, dx, vuw["ul"], vuw["ur"], mu["mul"], mu["mur"],
                                                 rho, -Fp(1.0))

                # west face
                if i == 0:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i - 1, j, k])
                a_W = calculate_face_coefficient(area_x, dx, vuw["ul"], vuw["ur"], mu["mul"], mu["mur"],
                                                 rho, Fp(1.0))

                # north face
                if j == n_y_cell - 1:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i, j + 1, k])
                a_N = calculate_face_coefficient(area_y, dy, vuw["vl"], vuw["vr"], mu["mul"], mu["mur"],
                                                 rho, -Fp(1.0))

                # south face
                if j == 0:
                    rho = density[i, j, k]
                else:
                    rho = Fp(0.5) * (density[i, j, k] + density[i, j - 1, k])
                a_S = calculate_face_coefficient(area_y, dy, vuw["vl"], vuw["vr"], mu["mul"], mu["mur"],
                                                 rho, Fp(1.0))

                if dim == 3:
                    # top face
                    if k == n_z_cell - 1:
                        rho = density[i, j, k]
                    else:
                        rho = Fp(0.5) * (density[i, j, k] + density[i, j, k + 1])
                    a_T = calculate_face_coefficient(area_z, dz, vuw["wl"], vuw["wr"], mu["mul"],
                                                     mu["mur"], rho, -Fp(1.0))

                    # bottom face
                    if k == 0:
                        rho = density[i, j, k]
                    else:
                        rho = Fp(0.5) * (density[i, j, k] + density[i, j, k - 1])
                    a_B = calculate_face_coefficient(area_z, dz, vuw["wl"], vuw["wr"], mu["mul"],
                                                     mu["mur"], rho, Fp(1.0))
                # 求解ap系数及源项
                rho = density[i, j, k]
                a_P0 = rho * vol / dt
                a_P = a_E + a_W + a_S + a_N + a_T + a_B + a_P0 - s_P * vol
                b = s_C * vol + a_P0 * t[i, j, k]
                # 存储数据
                t_coefficient[i, j, k, mesh_coefficient.aP_id.value] = a_P
                t_coefficient[i, j, k, mesh_coefficient.aW_id.value] = -a_W
                t_coefficient[i, j, k, mesh_coefficient.aE_id.value] = -a_E
                t_coefficient[i, j, k, mesh_coefficient.aS_id.value] = -a_S
                t_coefficient[i, j, k, mesh_coefficient.aN_id.value] = -a_N
                if dim == 3:
                    t_coefficient[i, j, k, mesh_coefficient.aT_id.value] = -a_T
                    t_coefficient[i, j, k, mesh_coefficient.aB_id.value] = -a_B
                t_coefficient[i, j, k, mesh_coefficient.ab_id.value] = b


def solve_boundary_conductivity_coefficient(x_cell_centroid, y_cell_centroid, z_cell_centroid, face_boundary, face_id, dim, n_x_cell, n_y_cell, n_z_cell,
                            conductivity_coefficient, mesh_coefficient, x, y, z, t,t_coefficient, physics_boundary_condition, temperature_boundary_condition:TemperatureBoundaryCondition):

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
                    faces.append(face_boundary[i,j,k,enum_face_id.value])

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
                        [face_physics_boundary[0],face_temperature_boundary[0], mesh_coefficient.aE_id.value, face_temperature[0], face_flux[0], x[i + 1], x_cell_centroid[i]],
                        [face_physics_boundary[1],face_temperature_boundary[1], mesh_coefficient.aW_id.value, face_temperature[1], face_flux[1], x[i], x_cell_centroid[i - 1]],
                        [face_physics_boundary[2],face_temperature_boundary[2], mesh_coefficient.aN_id.value, face_temperature[2], face_flux[2], y[j + 1], y_cell_centroid[j]],
                        [face_physics_boundary[3],face_temperature_boundary[3], mesh_coefficient.aS_id.value, face_temperature[3], face_flux[3], y[j], y_cell_centroid[j - 1]]]
                if dim == 3:
                    all_coefficient_list.append([face_physics_boundary[4], face_temperature_boundary[4], mesh_coefficient.aT_id.value,
                         face_temperature[4], face_flux[4], z[k + 1], z_cell_centroid[k]])
                    all_coefficient_list.append(
                        [face_physics_boundary[5], face_temperature_boundary[5], mesh_coefficient.aB_id.value,
                         face_temperature[5], face_flux[5], z[k], z_cell_centroid[k - 1]])

                for index,value in enumerate(all_coefficient_list):
                    if value[0] == PhysicsBoundaryID.inlet:
                        t_coefficient[i, j, k, mesh_coefficient.aP_id.value] -= t_coefficient[i, j, k, value[2]]
                        t_coefficient[i, j, k, mesh_coefficient.ab_id.value] -= Fp(2.0) * t_coefficient[i, j, k, value[2]] * value[3]
                        t_coefficient[i, j, k, value[2]] = Fp(0.0)
                    elif value[0] == PhysicsBoundaryID.outlet:
                        t_coefficient[i, j, k, mesh_coefficient.aP_id.value] += t_coefficient[i, j, k, value[2]]
                        t_coefficient[i, j, k, value[2]] = Fp(0.0)
                    elif value[0] == PhysicsBoundaryID.wall:
                        if value[1] == TemperatureBoundaryID.constant:
                            t_coefficient[i, j, k, mesh_coefficient.aP_id.value] -= t_coefficient[i, j, k, value[2]]
                            t_coefficient[i, j, k, mesh_coefficient.ab_id.value] -= Fp(2.0) * t_coefficient[i, j, k, value[2]] * value[3]
                            t_coefficient[i, j, k, value[2]] = Fp(0.0)
                        elif value[1] == TemperatureBoundaryID.heat_flux:
                            t_coefficient[i, j, k, mesh_coefficient.aP_id.value] += t_coefficient[i, j, k, value[2]]
                            t_coefficient[i, j, k, mesh_coefficient.ab_id.value] += t_coefficient[i, j, k, value[2]] * value[4] * Fp(2.0) * (value[5] - value[6]) / conductivity_coefficient
                            t_coefficient[i, j, k, value[2]] = Fp(0.0)
