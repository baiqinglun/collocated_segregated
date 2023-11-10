from case import CaseManager
from mesh import MeshManager
from fluid import Fluid
from fp import Fp
from boundary import FluidBoundaryCondition
from solve import SolveManager,ConvectionScheme
from post_process import PostProcessManager
from collocated_segregated import CollocatedSegregated
from read_setting import ReadSetting
from draw import DrawCurves

if __name__ == '__main__':
    # 读取配置文件
    readSetting = ReadSetting("setting.json")
    dim = readSetting.dim
    x_cell = readSetting.x_cell
    y_cell = readSetting.y_cell
    z_cell = readSetting.z_cell
    coordinate_limit_count = readSetting.coordinate_limit_count
    init_temperature = readSetting.init_temperature
    density = readSetting.density
    mu = readSetting.mu
    specific_heat_capacity = readSetting.specific_heat_capacity
    conductivity_coefficient = readSetting.conductivity_coefficient
    x_min_type = readSetting.x_min_type
    x_max_type = readSetting.x_max_type
    y_min_type = readSetting.y_min_type
    y_max_type = readSetting.y_max_type
    z_min_type = readSetting.z_min_type
    z_max_type = readSetting.z_max_type
    x_min_temperature_type = readSetting.x_min_temperature_type
    x_min_temperature_value = readSetting.x_min_temperature_value
    x_max_temperature_type = readSetting.x_max_temperature_type
    x_max_temperature_value = readSetting.x_max_temperature_value
    y_min_temperature_type = readSetting.y_min_temperature_type
    y_min_temperature_value = readSetting.y_min_temperature_value
    y_max_temperature_type = readSetting.y_max_temperature_type
    y_max_temperature_value = readSetting.y_max_temperature_value
    z_min_temperature_type = readSetting.z_min_temperature_type
    z_min_temperature_value = readSetting.z_min_temperature_value
    z_max_temperature_type = readSetting.z_max_temperature_type
    z_max_temperature_value = readSetting.z_max_temperature_value
    iter_step_count = readSetting.iter_step_count
    solve_equation_step_count = readSetting.solve_equation_step_count
    relax_factor = readSetting.relax_factor
    solve_equation_tolerance = readSetting.solve_equation_tolerance
    residual_error = readSetting.residual_error
    save_residual_frequency = readSetting.save_residual_frequency
    output_frequency = readSetting.output_frequency
    output_folder = readSetting.output_folder
    linear_equation_residual_filename = readSetting.linear_equation_residual_filename
    nonlinear_equation_residual_filename = readSetting.nonlinear_equation_residual_filename
    vtk_data_filename = readSetting.vtk_data_filename
    dat_filename = readSetting.dat_filename
    is_test = readSetting.is_test
    is_show_figure = readSetting.is_show_figure
    u = readSetting.u

    if not is_test:
        # 网格划分
        meshManager = MeshManager(dim, x_cell, y_cell, z_cell)
        coordinate_limit_count[0] = Fp(-0.5/(x_cell-1))
        coordinate_limit_count[1] = Fp(1.0+0.5/(x_cell-1))
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
        fluidBoundary.create_boundary(dim, x_min_type, x_max_type, y_min_type, y_max_type)
        # fluidBoundary.create_boundary_temperature(dim,
        #                                           x_min_temperature_type,  Fp(x_min_temperature_value),
        #                                           x_max_temperature_type,  Fp(x_max_temperature_value),
        #                                           y_min_temperature_value, Fp(y_min_temperature_value),
        #                                           y_max_temperature_value, Fp(y_max_temperature_value))
        if y_cell == 1:
            fluidBoundary.create_boundary_temperature(dim,
                                                  'constant', Fp(0.0),  # xmin
                                                  'constant', Fp(500.0),  # xmax
                                                  'heat_flux', Fp(0.0),  # ymin
                                                  'heat_flux', Fp(0.0))  # ymax
        elif x_cell == 1:
            fluidBoundary.create_boundary_temperature(dim,
                                                 'heat_flux', Fp(0.0),  # xmin
                                                 'heat_flux', Fp(0.0),  # xmax
                                                 'constant', Fp(0.0),  # ymin
                                                 'constant', Fp(1.0))  # ymax
        # 求解
        solveManager = SolveManager()
        solveManager.set_iter_step_count(iter_step_count)
        solveManager.set_solve_equation_step_count(solve_equation_step_count)
        solveManager.set_relax_factor(relax_factor)
        solveManager.set_solve_equation_tolerance(solve_equation_tolerance)
        solveManager.set_residual_error(residual_error)
        solveManager.set_convection_scheme(ConvectionScheme.cd)

        # 后处理
        postProcessManager = PostProcessManager(output_folder)
        postProcessManager.set_frequency(save_residual_frequency, output_frequency)
        postProcessManager.set_output_files_name(vtk_data_filename,dat_filename,linear_equation_residual_filename,nonlinear_equation_residual_filename)
        postProcessManager.write_vtk_file(meshManager, caseManager)
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


