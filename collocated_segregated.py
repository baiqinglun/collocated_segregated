import numpy as np
from fp import Fp
from mesh import MeshManager
from solve import SolveManager, EquationType
from post_process import PostProcessManager
from case import CaseManager, MeshCoefficient2D, MeshCoefficient3D
from fluid import Fluid
from collocated_sharing import solve_conduction_coefficient, solve_boundary_conductivity_coefficient, \
    solve_flow_conduction_coefficient, solve_boundary_flow_conductivity_coefficient
from boundary import FluidBoundaryCondition
from solve_linear_equation import scalar_Pj, eqn_scalar_norm2
import time
from draw import DrawCurves

"""
@name: CollocatedSegregated
@description: 分离求解器
@variable:
    mesh：网格信息
    case：数据信息
    solve：求解器信息
    boundary：边界条件
    post：后处理信息
    fluid：流体信息
    drawer：绘图信息
@function: 
    circulate()：循环求解
    solve_conduction()：求解扩散方程
    solve_flow()：求解对流方程
    solve_conduction_flow()：求解对流扩散方程
    create_boundary_temperature()：创建温度边界
"""


class CollocatedSegregated:
    def __init__(self, mesh: MeshManager, case: CaseManager, solve: SolveManager, boundary: FluidBoundaryCondition,
                 post: PostProcessManager, fluid: Fluid, drawer: DrawCurves = None):
        self.mesh = mesh
        self.dim = mesh.dim
        self.n_x_cell = mesh.n_x_cell
        self.n_y_cell = mesh.n_y_cell
        self.n_z_cell = mesh.n_z_cell
        self.n_x_point = mesh.n_x_point
        self.mesh_coefficient_count = case.mesh_coefficient.count.value  # 矩阵系数的个数 # aP,aW,aE,aS,aN,b
        self.x = mesh.x
        self.y = mesh.y
        self.z = mesh.z
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.dz = mesh.dz
        self.x_cell_centroid = mesh.x_cell_centroid
        self.y_cell_centroid = mesh.y_cell_centroid
        self.z_cell_centroid = mesh.z_cell_centroid

        self.case = case
        self.t = case.t
        self.u = case.u
        self.v = case.v
        self.w = case.w
        self.t_coefficient = case.t_coefficient
        self.mesh_coefficient = case.mesh_coefficient
        self.initial_u = case.initial_u

        self.solve = solve
        self.iter_step_count = solve.iter_step_count
        self.dt = solve.dt
        self.equation_type = solve.equation_type  # 用到的方程
        self.solve_equation_step_count = solve.solve_equation_step_count
        self.solve_equation_tolerance = solve.solve_equation_tolerance
        self.relax_factor = solve.relax_factor
        self.residual_error = solve.residual_error
        self.solve_equation_count = solve.solve_equation_count
        self.is_finish = solve.is_finish
        self.convection_scheme = solve.convection_scheme

        self.face_boundary = boundary.face_boundary
        self.face_id = boundary.face_id
        self.physics_boundary_condition = boundary.physics_boundary_condition
        self.temperature_boundary_condition = boundary.temperature_boundary_condition

        self.fluid = fluid
        self.density = fluid.density
        self.mu = fluid.mu
        self.conductivity_coefficient = fluid.conductivity_coefficient
        self.specific_heat_capacity = fluid.specific_heat_capacity
        self.source_term = fluid.source_term

        self.post = post
        self.save_output_frequency = post.save_output_frequency
        self.output_folder = post.output_folder

        self.drawer = drawer

        # 根据函数类型，调用不同的函数
        self.function_mapping = {
            EquationType.conduction: self.solve_conduction,
            EquationType.conduction_flow: self.solve_conduction_flow,
        }

    def solve_conduction(self, current_iter):
        """
        先求系数再考虑边界条件
        优点是逻辑清晰，考虑边界条件时，只需要在原系数进行修改
        :return:
        """
        # 先求系数
        solve_conduction_coefficient(self.dx, self.dy, self.dz, self.t_coefficient, self.mesh_coefficient, self.dim,
                                     self.n_x_cell, self.n_y_cell, self.n_z_cell,
                                     self.dt, self.specific_heat_capacity, self.conductivity_coefficient,
                                     self.source_term,
                                     self.density, self.t, self.u, self.v, self.w, self.solve.conduction_scheme)

        # 再考虑边界条件
        solve_boundary_conductivity_coefficient(self.x_cell_centroid, self.y_cell_centroid, self.z_cell_centroid,
                                                self.face_boundary, self.face_id, self.dim, self.n_x_cell,
                                                self.n_y_cell, self.n_z_cell,
                                                self.conductivity_coefficient, self.mesh_coefficient, self.x, self.y,
                                                self.z, self.t, self.t_coefficient, self.physics_boundary_condition,
                                                self.temperature_boundary_condition)
        # 是否初始化为0
        init_zero = False
        old_t = np.copy(self.t)
        scalar_Pj(self.dim, self.solve, self.post, current_iter, self.relax_factor, self.n_x_cell, self.n_y_cell,
                  self.n_z_cell, self.t_coefficient, self.t, init_zero, self.mesh_coefficient)
        # 计算旧温度和新温度的误差
        (self.solve.l2_t, self.solve.l2_max_t) = eqn_scalar_norm2(self.solve, self.dim, current_iter, self.n_x_cell,
                                                                  self.n_y_cell, self.n_z_cell, old_t, self.t,
                                                                  'temperature')
        if self.solve.l2_t / self.solve.l2_max_t < self.residual_error:
            self.solve.is_finish = True
            print('')
            print('----------------------------------------')
            print('Final iter = ', current_iter)
            print('it, l2_t/l2_max_t', current_iter, self.solve.l2_t / self.solve.l2_max_t)
            print('----------------------------------------')

    def solve_conduction_flow(self, current_iter):
        """
                先求系数再考虑边界条件
                优点是逻辑清晰，考虑边界条件时，只需要在原系数进行修改
                :return:
                """
        # 先求系数

        solve_flow_conduction_coefficient(self.dx, self.dy, self.dz, self.t_coefficient, self.mesh_coefficient,
                                          self.dim,
                                          self.n_x_cell, self.n_y_cell, self.n_z_cell,
                                          self.dt, self.specific_heat_capacity, self.conductivity_coefficient,
                                          self.source_term,
                                          self.density, self.t, self.u, self.v, self.w, self.solve.convection_scheme)
        # 再考虑边界条件
        solve_boundary_flow_conductivity_coefficient(self.x_cell_centroid, self.y_cell_centroid, self.z_cell_centroid,
                                                     self.face_boundary, self.face_id, self.dim, self.n_x_cell,
                                                     self.n_y_cell, self.n_z_cell,
                                                     self.conductivity_coefficient, self.mesh_coefficient, self.x,
                                                     self.y,
                                                     self.z, self.t, self.t_coefficient,
                                                     self.physics_boundary_condition,
                                                     self.temperature_boundary_condition)
        # 是否初始化为0
        init_zero = False
        old_t = np.copy(self.t)
        scalar_Pj(self.dim, self.solve, self.post, current_iter, self.relax_factor, self.n_x_cell, self.n_y_cell,
                  self.n_z_cell, self.t_coefficient, self.t, init_zero, self.mesh_coefficient)
        # 计算旧温度和新温度的误差
        (self.solve.l2_t, self.solve.l2_max_t) = eqn_scalar_norm2(self.solve, self.dim, current_iter, self.n_x_cell,
                                                                  self.n_y_cell, self.n_z_cell, old_t, self.t,
                                                                  'temperature')
        if self.solve.l2_t / self.solve.l2_max_t < self.residual_error:
            self.solve.is_finish = True
            print('')
            print('----------------------------------------')
            print('Final iter = ', current_iter)
            print('it, l2_t/l2_max_t', current_iter, self.solve.l2_t / self.solve.l2_max_t)
            print('----------------------------------------')

    def circulate(self):
        if self.equation_type in self.function_mapping:
            solve_function = self.function_mapping[self.equation_type]
            self.post.start_time = time.perf_counter()
            for current_iter in range(1, self.iter_step_count + 1):
                # 在第1步、每隔20步、最后一步输出
                if current_iter % 2 == 0 or current_iter == 1 or current_iter == self.iter_step_count:
                    print('')
                    print('---------------------------')
                    print('Begin iter = ', current_iter)
                    print('---------------------------')
                solve_function(current_iter)
                # 每隔多少步保存文件
                if current_iter % self.post.save_output_frequency == 0 or current_iter == self.iter_step_count or self.solve.is_finish:
                    self.post.write_vtk_file(self.mesh, self.case)
                    self.post.write_dat_file(self.mesh, self.case)
                # 输出残差文件并绘制残差
                if current_iter == 2 or current_iter % self.save_output_frequency == 0 or current_iter == self.iter_step_count or self.solve.is_finish:
                    print('it, total time, l2_t/l2_max_t', current_iter, time.perf_counter() - self.post.start_time,
                          self.solve.l2_t / self.solve.l2_max_t)

                    with open(f"{self.output_folder}/{self.post.nonlinear_equation_residual_filename}",
                              'a') as nonlinear_equation_residual_filename_id:  # 追加模式
                        nonlinear_equation_residual_filename_id.write(
                            "{}  {} {}\n".format(current_iter, time.perf_counter() - self.post.start_time,
                                                 self.solve.l2_t / self.solve.l2_max_t))
                if self.drawer:
                    self.drawer.draw(time.perf_counter() - self.post.start_time,
                                     [self.solve.l2_t / self.solve.l2_max_t])
            self.post.write_temperature_Pe_L_center(self.n_x_cell, self.t, self.initial_u, self.convection_scheme,
                                                    self.conductivity_coefficient)
            self.post.end_time = time.perf_counter()
            print("Total time", self.post.end_time - self.post.start_time)
