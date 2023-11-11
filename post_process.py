from mesh import MeshManager
from case import CaseManager
import numpy as np
from fp import Fp
from fluid import Fluid
from solve import SolveManager,ConvectionScheme
import math

class PostProcessManager:
    def __init__(self,output_folder):
        self.output_folder = output_folder
        self.linear_equation_residual_filename = "linear_equation_residual.res"
        self.nonlinear_equation_residual_filename = "nonlinear_equation_residual.res"
        self.vtk_data_filename = "data.vtk"
        self.dat_filename = "temperature_line.dat"

        self.save_residual_frequency = 20
        self.save_output_frequency = 20
        self.start_time = Fp(0.0)
        self.end_time = Fp(0.0)

    def set_linear_equation_residual_filename(self,linear_equation_residual_filename):
        self.linear_equation_residual_filename = linear_equation_residual_filename

    def set_nonlinear_equation_residual_filename(self,nonlinear_equation_residual_filename):
        self.nonlinear_equation_residual_filename = nonlinear_equation_residual_filename

    def set_vtk_data_filename(self,vtk_data_filename):
        self.vtk_data_filename = vtk_data_filename

    def set_dat_filename(self,dat_filename):
        self.dat_filename = dat_filename

    def set_save_residual_frequency(self,save_residual_frequency):
        self.save_residual_frequency = save_residual_frequency

    def set_output_frequency(self, output_frequency):
        self.save_output_frequency = output_frequency

    def set_frequency(self,save_residual_frequency,output_frequency):
        self.save_residual_frequency = save_residual_frequency
        self.save_output_frequency = output_frequency

    def set_output_files_name(self,vtk_data_filename,dat_filename,linear_equation_residual_filename,nonlinear_equation_residual_filename):
        self.linear_equation_residual_filename = linear_equation_residual_filename
        self.nonlinear_equation_residual_filename = nonlinear_equation_residual_filename
        self.vtk_data_filename = vtk_data_filename
        self.dat_filename = dat_filename

    def write_vtk_file(self,mesh:MeshManager,case:CaseManager):
        n_x_point = mesh.n_x_point
        n_y_point = mesh.n_y_point
        n_z_point = mesh.n_z_point
        x = mesh.x
        y = mesh.y
        z = mesh.z
        t = case.t

        with open(f"{self.output_folder}/{self.vtk_data_filename}", 'w') as vtk_fid:
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

    def write_dat_file(self,mesh:MeshManager,case:CaseManager):
        n_x_point = mesh.n_x_point
        n_y_point = mesh.n_y_point
        n_z_point = mesh.n_z_point
        x = mesh.x
        y = mesh.y
        z = mesh.z
        t = case.t

        with open(f"{self.output_folder}/{self.dat_filename}", 'w') as file:
            # Write temperature data along y-line
            i1 = int(0.0833 / 0.833 * n_x_point)  # x = 0.0833
            i2 = int(Fp(0.5 / 0.833) * n_x_point)  # x = 0.5
            k = 0
            for j in range(n_y_point - 1):
                file.write(f"{(y[j] + y[j + 1]) * 0.5} {t[i1, j, k]} {t[i2, j, k]}\n")

    # res线性残差文件写入
    def write_linear_residual_file(self):
        with open(f"{self.output_folder}/{self.linear_equation_residual_filename}", 'w') as linear_equation_residual_id:
            linear_equation_residual_id.write("#it_nl, it, tot_it, norm, init, max, rel\n")
            linear_equation_residual_id.close()

    # res非线性残差文件写入
    def write_nonlinear_equation_residual_filename(self):
        with open(f"{self.output_folder}/{self.nonlinear_equation_residual_filename}", 'w') as nonlinear_equation_residual_id:
            nonlinear_equation_residual_id.write("#it, walltime, l2_t/l2_max_t\n")
            nonlinear_equation_residual_id.close()

    def write_temperature_Pe_L_center(self, n_x_cell,t,initial_u,convection_scheme, conductivity_coefficient):
        # Build the local data for np array
        out_file = None
        if abs(conductivity_coefficient) > 10000:
            Pe_L = Fp(0.0)
        else:
            Pe_L = initial_u / conductivity_coefficient

        print("Check Pe_L ", Pe_L)

        # 0: Upwind; 1: CD; 2: Power-law; 3: SOU (to be implemented);

        print("Check conv_scheme ", convection_scheme)

        if convection_scheme == ConvectionScheme.upwind:
            out_file = f'{self.output_folder}/center_temp_x_upwind.dat'
        elif convection_scheme == ConvectionScheme.cd:
            out_file = f'{self.output_folder}/center_temp_x_center.dat'

        # Open temperature output files
        with open(out_file, 'a') as file3:
            # Write temperature data at center point
            i = int(n_x_cell / 2) - 1
            j = 0
            k = 0
            if abs(Pe_L) < Fp(1.e-3):
                file3.write(f"{Pe_L} {t[i, j, k]} {0.5}\n")
            else:
                xtemp = 0.5
                result = (math.exp(Pe_L * xtemp) - 1) / (math.exp(Pe_L) - 1)
                file3.write(f"{Pe_L} {t[i, j, k]} {result}\n")
