'''分离求解器'''
import time
import numpy as np
from mesh import MeshManager,Direction
from solve import SolveManager, EquationType
from post_process import PostProcessManager
from case import CaseManager
from fluid import Fluid
from collocated_sharing import solve_diffusion_coefficient,\
                                modify_diffusion_boundary_coefficient,\
                                solve_conduction_coefficient,\
                                modify_conduction_boundary_coefficient,\
                                solve_velocity_boundary,\
                                solve_momentum_coefficient,\
                                modify_momentum_boundary_coefficient,\
                                solve_momentum_gradp,\
                                rhie_chow_face_velocity,\
                                solve_pressure_coefficient,\
                                correct_pressure,\
                                correct_velocity
from boundary import FluidBoundaryCondition
from solve_linear_equation import scalar_Pj, eqn_scalar_norm2,eqn_scalar_norm2_p
from draw import DrawCurves

class CollocatedSegregated:
    """
    @name: CollocatedSegregated
    @description: 分离求解器
    @variable:
        mesh:网格信息
        case:数据信息
        solve:求解器信息
        boundary:边界条件
        post:后处理信息
        fluid:流体信息
        drawer:绘图信息
    @function: 
        circulate()：循环求解
        solve_conduction()：求解扩散方程
        solve_flow()：求解对流方程
        solve_conduction_flow()：求解对流扩散方程
        create_boundary_temperature()：创建温度边界
    """
    def __init__(self, mesh: MeshManager, case: CaseManager, solve: SolveManager,\
                 boundary: FluidBoundaryCondition,
                 post: PostProcessManager, fluid: Fluid, drawer: DrawCurves = None):
        # 模型相关
        self.mesh = mesh
        self.dim = mesh.dim
        self.n_x_cell = mesh.n_x_cell
        self.n_y_cell = mesh.n_y_cell
        self.n_z_cell = mesh.n_z_cell
        self.n_x_point = mesh.n_x_point
        self.mesh_coefficient_count = case.mesh_coefficient.COUNT.value
        self.x = mesh.x
        self.y = mesh.y
        self.z = mesh.z
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.dz = mesh.dz
        self.x_cell_centroid = mesh.x_cell_centroid
        self.y_cell_centroid = mesh.y_cell_centroid
        self.z_cell_centroid = mesh.z_cell_centroid

        # 数据相关
        self.case = case
        self.t = case.t
        self.p = case.p
        self.pp = case.pp
        self.u = case.u
        self.v = case.v
        self.w = case.w
        self.uf = case.uf
        self.vf = case.vf
        self.wf = case.wf
        self.old_u = case.old_u
        self.old_v = case.old_v
        self.old_w = case.old_w
        self.old_p = case.old_p
        self.old_t = case.old_t
        self.t_coefficient = case.t_coefficient
        self.p_coefficient = case.p_coefficient
        self.u_coefficient = case.u_coefficient
        self.v_coefficient = case.v_coefficient
        self.w_coefficient = case.w_coefficient
        self.mesh_coefficient = case.mesh_coefficient
        self.initial_u = case.initial_u
        
        # 求解器相关
        self.solve = solve
        self.dt = solve.dt
        self.equation_type = solve.equation_type #用到的方程
        self.solve_equation_step_count = solve.solve_equation_step_count
        self.solve_equation_tolerance = solve.solve_equation_tolerance
        self.solve_equation_count = solve.solve_equation_count
        self.is_finish = solve.is_finish
        self.conduction_scheme = solve.conduction_scheme
        self.diffusion_scheme = solve.diffusion_scheme
        self.relax_factor_t = solve.relax_factor_t
        self.relax_factor_p = solve.relax_factor_p
        self.relax_factor_u = solve.relax_factor_u
        self.relax_factor_v = solve.relax_factor_v
        self.relax_factor_w = solve.relax_factor_w
        self.residual_error_t = solve.residual_error_t
        self.residual_error_p = solve.residual_error_p
        self.residual_error_u = solve.residual_error_u
        self.residual_error_v = solve.residual_error_v
        self.residual_error_w = solve.residual_error_w
        self.mass_total = solve.mass_total
        self.temperature_total = solve.temperature_total
        self.iter_step_count = solve.iter_step_count

        # 边界条件相关
        self.p_outlet = boundary.p_outlet
        self.pp_outlet = boundary.pp_outlet
        self.face_boundary = boundary.face_boundary
        self.face_id = boundary.face_id
        self.physics_boundary_condition = boundary.physics_boundary_condition
        self.temperature_boundary_condition = boundary.temperature_boundary_condition
        self.velocity_boundary_condition = boundary.velocity_boundary_condition

        # 流体相关
        self.fluid = fluid
        self.density = fluid.density
        self.mu = fluid.mu
        self.conductivity_coefficient = fluid.conductivity_coefficient
        self.specific_heat_capacity = fluid.specific_heat_capacity
        self.source_term = fluid.source_term

        # 后处理相关
        self.post = post
        self.save_output_frequency = post.save_output_frequency
        self.output_folder = post.output_folder

        # 绘图相关
        self.drawer = drawer

        # 根据函数类型，调用不同的函数
        self.function_mapping = {
            EquationType.DIFFUSION: self.solve_diffusion,
            EquationType.CONVECTION: self.solve_convection,
            EquationType.DIFFUSION_CONVECTION: self.solve_conduction_flow,
        }

    def solve_diffusion(self, current_iter):
        """
        求解温度
        先求系数再考虑边界条件
        优点是逻辑清晰，考虑边界条件时，只需要在原系数进行修改
        :return:
        """
        # 先求系数
        solve_diffusion_coefficient(self.dx, self.dy, self.dz, self.t_coefficient, self.mesh_coefficient, self.dim,
                                     self.n_x_cell, self.n_y_cell, self.n_z_cell,
                                     self.dt, self.specific_heat_capacity, self.conductivity_coefficient,
                                     self.source_term,
                                     self.density, self.t, self.u, self.v, self.w, self.solve.conduction_scheme)

        # 再考虑边界条件
        modify_diffusion_boundary_coefficient(self.x_cell_centroid, self.y_cell_centroid, self.z_cell_centroid, \
                                              self.face_boundary, self.face_id,\
                                              self.dim, self.n_x_cell, self.n_y_cell, self.n_z_cell,\
                                              self.conductivity_coefficient, self.mesh_coefficient, \
                                              self.x, self.y, self.z, self.t, self.t_coefficient,\
                                              self.physics_boundary_condition,\
                                              self.temperature_boundary_condition)
        # 是否初始化为0
        init_zero = False
        old_t = np.copy(self.t)
        scalar_Pj(self.dim, self.solve, self.post, current_iter, self.relax_factor_t, self.n_x_cell, self.n_y_cell,
                  self.n_z_cell, self.t_coefficient, self.t, init_zero, self.mesh_coefficient,self.residual_error_t)
        # 计算旧温度和新温度的误差
        (self.solve.l2_t, self.solve.l2_max_t) = eqn_scalar_norm2(self.solve, self.dim, current_iter, self.n_x_cell,
                                                                  self.n_y_cell, self.n_z_cell, old_t, self.t,
                                                                  'temperature')
        if self.solve.l2_t / self.solve.l2_max_t < self.residual_error_t:
            self.solve.is_finish = True
            print('')
            print('----------------------------------------')
            print('Final iter = ', current_iter)
            print('it, l2_t/l2_max_t', current_iter, self.solve.l2_t / self.solve.l2_max_t)
            print('----------------------------------------')

    def solve_convection(self, current_iter):
        """
        求解温度场
        先求系数再考虑边界条件
        优点是逻辑清晰，考虑边界条件时，只需要在原系数进行修改
        :return:
        """
        # 先求系数

        solve_conduction_coefficient(self.dx, self.dy, self.dz, self.t_coefficient, self.mesh_coefficient,
                                          self.dim,
                                          self.n_x_cell, self.n_y_cell, self.n_z_cell,
                                          self.dt, self.specific_heat_capacity, self.conductivity_coefficient,
                                          self.source_term,
                                          self.density, self.t, self.u, self.v, self.w, self.conduction_scheme)
        # 再考虑边界条件
        modify_conduction_boundary_coefficient(self.x_cell_centroid, self.y_cell_centroid, self.z_cell_centroid,
                                                     self.face_boundary, self.face_id, self.dim, self.n_x_cell,
                                                     self.n_y_cell, self.n_z_cell,
                                                     self.conductivity_coefficient, self.mesh_coefficient, self.x,
                                                     self.y,
                                                     self.z,self.t_coefficient,
                                                     self.physics_boundary_condition,
                                                     self.temperature_boundary_condition)
        # 是否初始化为0
        init_zero = False
        old_t = np.copy(self.t)
        scalar_Pj(self.dim, self.solve, self.post, current_iter, self.relax_factor_t, self.n_x_cell, self.n_y_cell,
                  self.n_z_cell, self.t_coefficient, self.t, init_zero, self.mesh_coefficient,self.residual_error_t)
        # 计算旧温度和新温度的误差
        (self.solve.l2_t, self.solve.l2_max_t) = eqn_scalar_norm2(self.solve, self.dim, current_iter, self.n_x_cell,
                                                                  self.n_y_cell, self.n_z_cell, old_t, self.t,
                                                                  'temperature')
        if self.solve.l2_t / self.solve.l2_max_t < self.residual_error_t:
            self.solve.is_finish = True
            print('')
            print('----------------------------------------')
            print('Final iter = ', current_iter)
            print('it, l2_t/l2_max_t', current_iter, self.solve.l2_t / self.solve.l2_max_t)
            print('----------------------------------------')

    def solve_conduction_flow(self,current_iter):
        '''求解对流扩散'''
        solve_velocity_boundary(self.dim,self.uf,self.vf,self.wf,self.u,self.v,self.w,self.face_id,self.face_boundary,\
                            self.n_x_cell,self.n_y_cell,self.n_z_cell,self.physics_boundary_condition,\
                            self.velocity_boundary_condition)
        solve_momentum_coefficient(self.dim,self.dx,self.dy,self.dz,self.dt,self.conduction_scheme,\
                               self.mesh_coefficient,self.u_coefficient,self.v_coefficient,\
                               self.w_coefficient,self.n_x_cell,self.n_y_cell,self.n_z_cell,\
                               self.density,self.conductivity_coefficient,self.uf,self.vf,self.wf,self.u,self.v,self.w)
        
        modify_momentum_boundary_coefficient(self.dim,self.n_x_cell,self.n_y_cell,self.n_z_cell,\
                                         self.face_id,self.physics_boundary_condition,\
                                         self.velocity_boundary_condition,self.face_boundary,\
                                         self.u_coefficient,self.mesh_coefficient,Direction.X)
        modify_momentum_boundary_coefficient(self.dim,self.n_x_cell,self.n_y_cell,self.n_z_cell,\
                                         self.face_id,self.physics_boundary_condition,\
                                         self.velocity_boundary_condition,self.face_boundary,\
                                         self.u_coefficient,self.mesh_coefficient,Direction.Y)
        if self.dim == 3:
            modify_momentum_boundary_coefficient(self.dim,self.n_x_cell,self.n_y_cell,self.n_z_cell,\
                                            self.face_id,self.physics_boundary_condition,\
                                            self.velocity_boundary_condition,self.face_boundary,\
                                            self.u_coefficient,self.mesh_coefficient,Direction.Z)
        
        solve_momentum_gradp(self.dim,self.p_outlet,self.p,self.face_boundary,self.face_id,self.physics_boundary_condition,\
                         self.mesh_coefficient,self.n_x_cell,self.n_y_cell,self.n_z_cell,\
                         self.dx,self.dy,self.dz,self.u_coefficient,self.v_coefficient,self.w_coefficient)
        
        init_zero = False
        scalar_Pj(self.dim, self.solve, self.post, current_iter, self.relax_factor_u, self.n_x_cell, self.n_y_cell,
                  self.n_z_cell, self.u_coefficient, self.u, init_zero, self.mesh_coefficient,self.residual_error_u)
        scalar_Pj(self.dim, self.solve, self.post, current_iter, self.relax_factor_v, self.n_x_cell, self.n_y_cell,
                  self.n_z_cell, self.v_coefficient, self.v, init_zero, self.mesh_coefficient,self.residual_error_v)
        if self.dim == 3:
            scalar_Pj(self.dim, self.solve, self.post, current_iter, self.relax_factor_w, self.n_x_cell, self.n_y_cell,
                  self.n_z_cell, self.w_coefficient, self.w, init_zero, self.mesh_coefficient,self.residual_error_w)


        rhie_chow_face_velocity(self.dim,self.old_u,self.old_v,self.old_w,self.n_x_cell,self.n_y_cell,self.n_z_cell,\
                                self.dx,self.dy,self.dz,self.u,self.v,self.w,self.uf,self.vf,self.wf,\
                                self.p,self.u_coefficient,self.v_coefficient,self.w_coefficient,self.density,self.dt,\
                                self.face_boundary,self.face_id,self.mesh_coefficient,self.p_outlet,self.physics_boundary_condition)
        
        solve_velocity_boundary(self.dim,self.uf,self.vf,self.wf,self.u,self.v,self.w,self.face_id,self.face_boundary,\
                            self.n_x_cell,self.n_y_cell,self.n_z_cell,self.physics_boundary_condition,\
                            self.velocity_boundary_condition)

        solve_pressure_coefficient(self.dim,self.n_x_cell,self.n_y_cell,self.n_z_cell,self.dx,self.dy,self.dz,\
                               self.physics_boundary_condition,self.face_boundary,self.face_id,\
                               self.u_coefficient,self.v_coefficient,self.w_coefficient,self.density,\
                               self.mesh_coefficient,self.uf,self.vf,self.wf,self.p_coefficient)
        
        scalar_Pj(self.dim, self.solve, self.post, current_iter, self.relax_factor_p, self.n_x_cell, self.n_y_cell,
                  self.n_z_cell, self.p_coefficient, self.p, init_zero, self.mesh_coefficient,self.residual_error_p)
        
        (self.solve.l2_pp, self.solve.l2_max_pp) = eqn_scalar_norm2_p(self.solve, self.dim, current_iter,\
                                                                       self.n_x_cell, self.n_y_cell, self.n_z_cell,\
                                                                        self.p_coefficient,self.mesh_coefficient, 'p_coefficient')

        correct_pressure(self.n_x_cell, self.n_y_cell,self.n_z_cell, self.relax_factor_p, self.pp, self.p)

        correct_velocity(self.dim,self.pp_outlet,self.dx,self.dy,self.dz,self.n_x_cell,self.n_y_cell,self.n_z_cell,\
                        self.u,self.v,self.w,self.uf,self.vf,self.wf,self.u_coefficient,self.v_coefficient,self.w_coefficient,\
                        self.mesh_coefficient,self.face_id,self.face_boundary,self.physics_boundary_condition,\
                        self.pp)
        
        (self.solve.l2_u, self.solve.l2_max_u) = eqn_scalar_norm2(self.solve, self.dim, current_iter,\
                                                                  self.n_x_cell, self.n_y_cell, self.n_z_cell,\
                                                                  self.u,self.old_u, 'u')
        (self.solve.l2_v, self.solve.l2_max_v) = eqn_scalar_norm2(self.solve, self.dim, current_iter,\
                                                                  self.n_x_cell, self.n_y_cell, self.n_z_cell,\
                                                                  self.v,self.old_v, 'v')
        if self.dim == 3:
            (self.solve.l2_v, self.solve.l2_max_v) = eqn_scalar_norm2(self.solve, self.dim, current_iter,\
                                                                  self.n_x_cell, self.n_y_cell, self.n_z_cell,\
                                                                  self.w,self.old_w, 'w')

        if (self.solve.l2_u/self.solve.l2_max_u < self.mass_total and self.solve.l2_v/self.solve.l2_max_v < self.mass_total and \
           self.solve.l2_w/self.solve.l2_max_w < self.mass_total and self.solve.l2_p/self.solve.l2_max_p < self.mass_total) or \
           (self.solve.l2_pp/self.solve.l2_max_pp < self.mass_total):
            self.solve.is_finish = True
            print('')
            print('----------------------------')
            print('Final iter =', current_iter)
            print('it, l2_u/l2_max_u, l2_v/l2_max_v, l2_w/l2_max_w, l2_p/l2_max_p, l2_pp/l2_max_pp', \
                   current_iter,self.solve.l2_u/self.solve.l2_max_u, self.solve.l2_v/self.solve.l2_max_v,\
                    self.solve.l2_w/self.solve.l2_max_w, self.solve.l2_p/self.solve.l2_max_p, \
                       self.solve.l2_pp/self.solve.l2_max_pp)
            print('----------------------------')
 
    
    def circulate(self):
        '''
        循环计算
        '''
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
                self.solve_convection(current_iter)
                # 每隔多少步保存文件
                if current_iter % self.post.save_output_frequency == 0 or current_iter == self.iter_step_count or self.solve.is_finish:
                    self.post.write_pressure_vtk_file(self.mesh, self.case)
                    self.post.write_velocity_vtk_file(self.mesh, self.case)
                    self.post.write_temperature_vtk_file(self.mesh, self.case)
                    self.post.write_dat_file(self.mesh, self.case)
                # 输出残差文件并绘制残差
                if current_iter == 2 or current_iter % self.save_output_frequency == 0 or current_iter == self.iter_step_count or self.solve.is_finish:
                    print('it, total time, l2_t/l2_max_t', current_iter, time.perf_counter() - self.post.start_time,
                          self.solve.l2_t / self.solve.l2_max_t)

                    with open(f"{self.output_folder}/{self.post.nonlinear_equation_residual_filename}",
                              'a',encoding="utf-8") as nonlinear_equation_residual_filename_id:  # 追加模式
                        nonlinear_equation_residual_filename_id.write(
                            f"{current_iter} {time.perf_counter() - self.post.start_time} {self.solve.l2_t / self.solve.l2_max_t}\n")
                
                if self.drawer:
                    self.drawer.draw(time.perf_counter() - self.post.start_time,
                                     [self.solve.l2_t / self.solve.l2_max_t])
           # self.post.write_temperature_Pe_L_center(self.n_x_cell, self.t, self.initial_u, self.conduction_scheme,
           #                                         self.conductivity_coefficient)
            self.post.end_time = time.perf_counter()
            print("Total time", self.post.end_time - self.post.start_time)