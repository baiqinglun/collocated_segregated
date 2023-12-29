'''后处理'''
import math
import numpy as np
from mesh import MeshManager
from case import CaseManager
from fp import Fp
from solve import DiscreteScheme

class PostProcessManager:
    """
    @name: PostProcessManager
    @description: 定义后处理相关参数
    @variable:
        output_folder:输出文件夹
        linear_equation_residual_filename:方程求解残差文件名
        nonlinear_equation_residual_filename:残差文件名
        vtk_data_filename:vtk数据文件名
        dat_filename:温度
        save_residual_frequency:残差保存频率
        save_output_frequency:输出文件保存频率
        start_time:开始时间
        end_time:结束时间
    @function: 
        write_*():文件写入
    """
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.linear_equation_residual_filename = "linear_equation_residual.res"
        self.nonlinear_equation_residual_filename = "nonlinear_equation_residual.res"
        self.pressure_vtk_data_filename = "pressure.vtk"
        self.velocity_vtk_data_filename = "velocity.vtk"
        self.temperature_vtk_data_filename = "temperature.vtk"
        self.dat_filename = "temperature_line.dat"

        self.save_residual_frequency = 20
        self.save_output_frequency = 20
        self.start_time = Fp(0.0)
        self.end_time = Fp(0.0)

    def set_linear_equation_residual_filename(self, linear_equation_residual_filename):
        '''设置线性残差文件名'''
        self.linear_equation_residual_filename = linear_equation_residual_filename

    def set_nonlinear_equation_residual_filename(self, nonlinear_equation_residual_filename):
        '''设置非线性残差文件名'''
        self.nonlinear_equation_residual_filename = nonlinear_equation_residual_filename

    def set_pressure_vtk_data_filename(self, pressure_vtk_data_filename):
        '''设置pressure vtk数据文件名'''
        self.pressure_vtk_data_filename = pressure_vtk_data_filename

    def set_velocity_vtk_data_filename(self, velocity_vtk_data_filename):
        '''设置velocity vtk数据文件名'''
        self.velocity_vtk_data_filename = velocity_vtk_data_filename
        
    def set_temperature_vtk_data_filename(self, temperature_vtk_data_filename):
        '''设置temperature vtk数据文件名'''
        self.temperature_vtk_data_filename = temperature_vtk_data_filename

    def set_dat_filename(self, dat_filename):
        '''设置dat数据文件名'''
        self.dat_filename = dat_filename

    def set_save_residual_frequency(self, save_residual_frequency):
        '''设置保存频率'''
        self.save_residual_frequency = save_residual_frequency

    def set_output_frequency(self, output_frequency):
        '''设置输出频率'''
        self.save_output_frequency = output_frequency

    def set_frequency(self, save_residual_frequency, output_frequency):
        '''设置残差频率、输出频率'''
        self.save_residual_frequency = save_residual_frequency
        self.save_output_frequency = output_frequency

    def set_output_files_name(self, pressure_vtk_data_filename, temperature_vtk_data_filename,\
                              velocity_vtk_data_filename,dat_filename, \
                              linear_equation_residual_filename,nonlinear_equation_residual_filename):
        '''设置输出文件名'''
        self.linear_equation_residual_filename = linear_equation_residual_filename
        self.nonlinear_equation_residual_filename = nonlinear_equation_residual_filename
        self.pressure_vtk_data_filename = pressure_vtk_data_filename
        self.temperature_vtk_data_filename = temperature_vtk_data_filename
        self.velocity_vtk_data_filename = velocity_vtk_data_filename
        self.dat_filename = dat_filename

    def write_temperature_vtk_file(self, mesh: MeshManager, case: CaseManager):
        '''写入vtk文件'''
        n_x_point = mesh.n_x_point
        n_y_point = mesh.n_y_point
        n_z_point = mesh.n_z_point
        x = mesh.x
        y = mesh.y
        z = mesh.z
        t = case.t

        with open(file=f"{self.output_folder}/{self.temperature_vtk_data_filename}", mode='w',encoding="utf-8") as vtk_fid:
            vtk_fid.write('# vtk DataFile Version 3.0\n')
            vtk_fid.write('flash 3d grid and solution\n')
            vtk_fid.write('ASCII\n')
            vtk_fid.write('DATASET RECTILINEAR_GRID\n')

            vtk_fid.write(f"DIMENSIONS {n_x_point} {n_y_point} {n_z_point}\n")
            vtk_fid.write(f"X_COORDINATES {n_x_point} float\n")
            vtk_fid.write(' '.join(str(i) for i in x) + '\n')
            vtk_fid.write(f"Y_COORDINATES {n_y_point} float\n")
            vtk_fid.write(' '.join(str(i) for i in y) + '\n')
            vtk_fid.write(f"Z_COORDINATES {n_z_point} float\n")
            vtk_fid.write(' '.join(str(i) for i in z) + '\n')

            n_cell = (n_x_point - 1) * (n_y_point - 1) * (n_z_point - 1)
            vtk_fid.write(f"CELL_DATA {n_cell}\n")

            vtk_fid.write('{:s}'.format("FIELD FieldData 1\n"))  # {:s}格式化字符串的占位符

            vtk_fid.write(f"t 1 {n_cell} float\n")

            t_arr = np.ravel(t[:, :, :], order='F')  # 将多维数组转化为一维数组，且不会产生源数据的副本
            vtk_fid.write(' '.join(str(i) for i in t_arr) + '\n')  # 将t_arr 的可迭代对象中的元素使用' '连接成一个字符串

    def write_pressure_vtk_file(self, mesh: MeshManager, case: CaseManager):
        '''写入vtk文件'''
        n_x_point = mesh.n_x_point
        n_y_point = mesh.n_y_point
        n_z_point = mesh.n_z_point
        x = mesh.x
        y = mesh.y
        z = mesh.z
        p = case.p

        with open(file=f"{self.output_folder}/{self.pressure_vtk_data_filename}", mode='w',encoding="utf-8") as vtk_fid:
            vtk_fid.write('# vtk DataFile Version 3.0\n')
            vtk_fid.write('flash 3d grid and solution\n')
            vtk_fid.write('ASCII\n')
            vtk_fid.write('DATASET RECTILINEAR_GRID\n')

            vtk_fid.write(f"DIMENSIONS {n_x_point} {n_y_point} {n_z_point}\n")
            vtk_fid.write(f"X_COORDINATES {n_x_point} float\n")
            vtk_fid.write(' '.join(str(i) for i in x) + '\n')
            vtk_fid.write(f"Y_COORDINATES {n_y_point} float\n")
            vtk_fid.write(' '.join(str(i) for i in y) + '\n')
            vtk_fid.write(f"Z_COORDINATES {n_z_point} float\n")
            vtk_fid.write(' '.join(str(i) for i in z) + '\n')

            n_cell = (n_x_point - 1) * (n_y_point - 1) * (n_z_point - 1)
            vtk_fid.write(f"CELL_DATA {n_cell}\n")

            vtk_fid.write('{:s}'.format("FIELD FieldData 1\n"))  # {:s}格式化字符串的占位符

            vtk_fid.write(f"p 1 {n_cell} float\n")

            p_arr = np.ravel(p[:, :, :], order='F')  # 将多维数组转化为一维数组，且不会产生源数据的副本
            vtk_fid.write(' '.join(str(i) for i in p_arr) + '\n')  # 将t_arr 的可迭代对象中的元素使用' '连接成一个字符串

    def write_velocity_vtk_file(self, mesh: MeshManager, case: CaseManager):
        '''写入vtk文件'''
        n_x_point = mesh.n_x_point
        n_y_point = mesh.n_y_point
        n_z_point = mesh.n_z_point
        x = mesh.x
        y = mesh.y
        z = mesh.z
        u = case.u
        v = case.v
        w = case.w

        with open(file=f"{self.output_folder}/{self.velocity_vtk_data_filename}", mode='w',encoding="utf-8") as vtk_fid:
            vtk_fid.write('# vtk DataFile Version 3.0\n')
            vtk_fid.write('flash 3d grid and solution\n')
            vtk_fid.write('ASCII\n')
            vtk_fid.write('DATASET RECTILINEAR_GRID\n')

            vtk_fid.write(f"DIMENSIONS {n_x_point} {n_y_point} {n_z_point}\n")
            vtk_fid.write(f"X_COORDINATES {n_x_point} float\n")
            vtk_fid.write(' '.join(str(i) for i in x) + '\n')
            vtk_fid.write(f"Y_COORDINATES {n_y_point} float\n")
            vtk_fid.write(' '.join(str(i) for i in y) + '\n')
            vtk_fid.write(f"Z_COORDINATES {n_z_point} float\n")
            vtk_fid.write(' '.join(str(i) for i in z) + '\n')

            n_cell = (n_x_point - 1) * (n_y_point - 1) * (n_z_point - 1)
            vtk_fid.write(f"CELL_DATA {n_cell}\n")

            vtk_fid.write('{:s}'.format("FIELD FieldData 3\n"))  # {:s}格式化字符串的占位符
            for var in ['u','v','w']:
                vtk_fid.write(f"{var} 1 {n_cell} float\n")
                var_arr = np.ravel(eval(f"{var}")[:, :, :], order='F')  # 将多维数组转化为一维数组，且不会产生源数据的副本
                vtk_fid.write(' '.join(str(i) for i in var_arr) + '\n')  # 将t_arr 的可迭代对象中的元素使用' '连接成一个字符串

    def write_dat_file(self, mesh: MeshManager, case: CaseManager):
        '''写入dat文件'''
        n_x_point = mesh.n_x_point
        n_y_point = mesh.n_y_point
        n_z_point = mesh.n_z_point
        x = mesh.x
        y = mesh.y
        z = mesh.z
        t = case.t

        with open(file=f"{self.output_folder}/{self.dat_filename}",mode='w',encoding='utf-8') as file:
            # Write temperature data along y-line
            i1 = int(0.0833 / 0.833 * n_x_point)  # x = 0.0833
            # i2 = int(Fp(0.5 / 0.833) * n_x_point)  # x = 0.5
            k = 0
           
            for j in range(n_y_point - 1):
                # file.write(f"{(y[j] + y[j + 1]) * 0.5} {t[i1, j, k]} {t[i2, j, k]}\n")
                file.write(f"{(y[j] + y[j + 1]) * 0.5} {t[i1, j, k]} \n")

    def write_linear_residual_file(self):
        '''res线性残差文件写入'''
        with open(file=f"{self.output_folder}/{self.linear_equation_residual_filename}", mode='w',encoding='utf-8') as linear_equation_residual_id:
            linear_equation_residual_id.write("#it_nl, it, tot_it, norm, init, max, rel\n")
            linear_equation_residual_id.close()

    def write_nonlinear_equation_residual_filename(self):
        '''res非线性残差文件写入'''
        with open(file=f"{self.output_folder}/{self.nonlinear_equation_residual_filename}",
                  mode='w',encoding='utf-8') as nonlinear_equation_residual_id:
            nonlinear_equation_residual_id.write("#it, walltime, l2_t/l2_max_t\n")
            nonlinear_equation_residual_id.close()

    def write_temperature_Pe_L_center(self, n_x_cell, t, initial_u, convection_scheme, conductivity_coefficient):
        '''写入温度、PeL'''
        out_file = None
        if abs(conductivity_coefficient) > 10000:
            PeL = Fp(0.0)
        else:
            PeL = initial_u / conductivity_coefficient

        print("Check Pe_L ", PeL)

        # 0: Upwind; 1: CD; 2: Power-law; 3: SOU (to be implemented);
        print("Check conv_scheme ", convection_scheme)

        if convection_scheme == DiscreteScheme.UPWIND:
            out_file = f'{self.output_folder}/center_temp_x_upwind.dat'
        elif convection_scheme == DiscreteScheme.CD:
            out_file = f'{self.output_folder}/center_temp_x_center.dat'

        with open(file=out_file,mode='a',encoding='utf-8') as file3:
            # 模拟解，若为偶数4，t[1,0,0]，第二个网格，xtemp=1/3
            # 精确解，若为基数3，t[1,0,0]，第二个网格，xtemp=1/2
            i = int((n_x_cell + 1) / 2) - 1
            j = 0
            k = 0
            if abs(PeL) < Fp(1.e-3):
                file3.write(f"{PeL} {t[i, j, k]} {0.5}\n")
            else:
                xtemp = (n_x_cell / 2 - 1) / (n_x_cell - 1) if (n_x_cell % 2 == 0) else ((n_x_cell + 1) / 2 - 1) / (
                            n_x_cell - 1)
                result = (math.exp(PeL * xtemp) - 1) / (math.exp(PeL) - 1)
                file3.write(f"{PeL} {t[i, j, k]} {result}\n")
