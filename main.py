'''入口函数'''
from case import CaseManager
from mesh import MeshManager
from fluid import Fluid
from fp import Fp
from boundary import FluidBoundaryCondition
from solve import SolveManager,DiscreteScheme,EquationType
from post_process import PostProcessManager
from collocated_segregated import CollocatedSegregated
from read_setting import ReadSetting
from draw import DrawCurves

if __name__ == '__main__':
    # 读取配置文件
    readSetting = ReadSetting("setting.json")
    dim,x_cell,y_cell,z_cell,coordinate_limit_count = readSetting.get_mesh_settings()
    init_temperature,u = readSetting.get_case_settings()
    density,mu,specific_heat_capacity,conductivity_coefficient = readSetting.get_fluid_settingsa()
    x_min_type,x_max_type,y_min_type,y_max_type,z_min_type,\
    z_max_type,x_min_temperature_type,x_min_temperature_value,\
    x_max_temperature_type,x_max_temperature_value,\
    y_min_temperature_type,y_min_temperature_value,\
    y_max_temperature_type,y_max_temperature_value,\
    z_min_temperature_type,z_min_temperature_value,\
    z_max_temperature_type,z_max_temperature_value = readSetting.get_boundary_settings()
    iter_step_count,solve_equation_step_count,relax_factor,solve_equation_tolerance,residual_error  = readSetting.get_solve_settings()
    save_residual_frequency,output_frequency,output_folder,\
    linear_equation_residual_filename,nonlinear_equation_residual_filename,\
    pressure_vtk_data_filename,temperature_vtk_data_filename,velocity_vtk_data_filename,dat_filename= readSetting.get_post_settings()
    is_test,is_show_figure = readSetting.get_other_settings()

    if not is_test:
        # 网格划分
        meshManager = MeshManager(dim, x_cell, y_cell, z_cell)
        meshManager.create_coordinates(coordinate_limit_count)

        # 存储数据
        caseManager = CaseManager(meshManager)
        caseManager.create_mesh_data()
        caseManager.set_temperature(init_temperature)
        caseManager.create_mesh_coefficient()
        caseManager.set_u(u)

        # 定义流体
        fluid = Fluid(meshManager)
        fluid.set_density(Fp(density))
        fluid.set_physical_property(density=Fp(density), mu=Fp(mu), specific_heat_capacity=Fp(specific_heat_capacity),
                                    conductivity_coefficient=Fp(conductivity_coefficient))

        # 边界条件
        fluidBoundary = FluidBoundaryCondition(dim=dim)
        fluidBoundary.create_face_boundary(meshManager)
        print(x_min_type, x_max_type, y_min_type, y_max_type,z_min_type,z_max_type)
        fluidBoundary.create_boundary(dim, x_min_type, x_max_type, y_min_type, y_max_type,z_min_type,z_max_type)

        # 求解
        solveManager = SolveManager()
        solveManager.set_iter_step_count(iter_step_count)
        solveManager.set_solve_equation_step_count(solve_equation_step_count)
        solveManager.set_relax_factor(Fp(relax_factor))
        solveManager.set_solve_equation_tolerance(solve_equation_tolerance)
        solveManager.set_residual_error(residual_error)
        solveManager.set_diffusion_scheme(DiscreteScheme.CD)
        solveManager.set_conduction_scheme(DiscreteScheme.CD)
        solveManager.set_equation_type(EquationType.DIFFUSION_CONVECTION)

        fluidBoundary.create_boundary_temperature(dim,
                                                      x_min_temperature_type,  Fp(x_min_temperature_value),
                                                      x_max_temperature_type,  Fp(x_max_temperature_value),
                                                      y_min_temperature_value, Fp(y_min_temperature_value),
                                                      y_max_temperature_value, Fp(y_max_temperature_value))

        if solveManager.equation_type == EquationType.CONVECTION:
            fluidBoundary.create_boundary_temperature(dim,
                                                      x_min_temperature_type,  Fp(x_min_temperature_value),
                                                      x_max_temperature_type,  Fp(x_max_temperature_value),
                                                      y_min_temperature_value, Fp(y_min_temperature_value),
                                                      y_max_temperature_value, Fp(y_max_temperature_value))
        elif solveManager.equation_type == EquationType.DIFFUSION_CONVECTION:
            if y_cell == 1:
                fluidBoundary.create_boundary_temperature(dim,
                                                      'constant', Fp(0.0),  # xmin
                                                      'constant', Fp(1.0),  # xmax
                                                      'heat_flux', Fp(0.0),  # ymin
                                                      'heat_flux', Fp(0.0))  # ymax
            elif x_cell == 1:
                fluidBoundary.create_boundary_temperature(dim,
                                                     'heat_flux', Fp(0.0),  # xmin
                                                     'heat_flux', Fp(0.0),  # xmax
                                                     'constant', Fp(0.0),  # ymin
                                                     'constant', Fp(1.0))  # ymax
        fluidBoundary.create_boundary_velocity(dim,\
                                ['constant','constant','flux'], [Fp(1.0), Fp(0.0), Fp(0.0)],  # xmin
                                ['None','None','None'], [Fp(0.0), Fp(0.0), Fp(0.0)],  # xmax
                                ['constant','constant','flux'], [Fp(0.0), Fp(0.0), Fp(0.0)], # ymin
                                ['constant','constant','flux'], [Fp(0.0), Fp(0.0), Fp(0.0)], # ymax
                                ['constant','constant','flux'], [Fp(0.0), Fp(0.0), Fp(0.0)], # zmin
                                ['constant','constant','flux'], [Fp(0.0), Fp(0.0), Fp(0.0)]) # zmax

        # 后处理
        postProcessManager = PostProcessManager(output_folder)
        postProcessManager.set_frequency(save_residual_frequency, output_frequency)
        postProcessManager.set_output_files_name(pressure_vtk_data_filename,temperature_vtk_data_filename,velocity_vtk_data_filename,\
                                                 dat_filename,linear_equation_residual_filename,nonlinear_equation_residual_filename)
        postProcessManager.write_temperature_vtk_file(meshManager, caseManager)
        postProcessManager.write_pressure_vtk_file(meshManager, caseManager)
        postProcessManager.write_velocity_vtk_file(meshManager, caseManager)
        postProcessManager.write_dat_file(meshManager, caseManager)
        postProcessManager.write_nonlinear_equation_residual_filename()
        postProcessManager.write_linear_residual_file()

        # 绘图
        drawer = None
        if is_show_figure:
            drawer = DrawCurves()

        collocatedSegregated = CollocatedSegregated(case=caseManager, mesh=meshManager, solve=solveManager,
                                                    boundary=fluidBoundary, post=postProcessManager, fluid=fluid,drawer=drawer)
        collocatedSegregated.circulate()

        print(caseManager.t)