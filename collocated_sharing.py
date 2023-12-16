'''分离求解器公共函数'''
from fp import Fp
from tool import calculate_area_volume, calculate_face_coefficient
from boundary import FluidBoundaryCondition, PhysicsBoundaryCondition, TemperatureBoundaryCondition
from case import CaseManager, MeshCoefficient2D, MeshCoefficient3D
from boundary_id import PhysicsBoundaryID, TemperatureBoundaryID, BoundaryLimitID


def solve_conduction_coefficient(dx, dy, dz, t_coefficient, mesh_coefficient, dim,
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

def solve_boundary_conductivity_coefficient(x_cell_centroid, y_cell_centroid, z_cell_centroid, face_boundary, face_id,
                                            dim, n_x_cell, n_y_cell, n_z_cell,
                                            conductivity_coefficient, mesh_coefficient, x, y, z, t, t_coefficient,
                                            physics_boundary_condition,
                                            temperature_boundary_condition: TemperatureBoundaryCondition):
    '''求解考虑温度边界的系数'''
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



def solve_flow_conduction_coefficient(dx, dy, dz, t_coefficient, mesh_coefficient, dim,
                                      n_x_cell, n_y_cell, n_z_cell,
                                      dt, specific_heat_capacity, conductivity_coefficient, source_term,
                                      density, t, u, v, w, conv_scheme):
    '''求解对流项'''
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


def solve_boundary_flow_conductivity_coefficient(x_cell_centroid, y_cell_centroid, z_cell_centroid, face_boundary,
                                                 face_id, dim, n_x_cell, n_y_cell, n_z_cell,
                                                 conductivity_coefficient, mesh_coefficient, x, y, z, t, t_coefficient,
                                                 physics_boundary_condition,
                                                 temperature_boundary_condition: TemperatureBoundaryCondition):
    '''求解对流项系数'''
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