'''读取json配置文件'''
import json

class ReadSetting:
    '''读取配置类'''
    def __init__(self,setting_file_name):
        with open(setting_file_name, 'r', encoding='UTF-8') as settings_file:
            settings_data = json.load(settings_file)
            self.dim = settings_data["dim"]
            self.x_cell = settings_data["x_cell"]
            self.y_cell = settings_data["y_cell"]
            self.z_cell = settings_data["z_cell"]
            self.coordinate_limit_count = settings_data["coordinate_limit_count"]
            
            self.density = settings_data["density"]
            self.mu = settings_data["mu"]
            self.specific_heat_capacity = settings_data["specific_heat_capacity"]
            self.conductivity_coefficient = settings_data["conductivity_coefficient"]
            
            self.init_temperature = settings_data["init_temperature"]
            self.u = settings_data["u"]
            
            self.x_min_type = settings_data["x_min_type"]
            self.x_max_type = settings_data["x_max_type"]
            self.y_min_type = settings_data["y_min_type"]
            self.y_max_type = settings_data["y_max_type"]
            self.z_min_type = settings_data["z_min_type"]
            self.z_max_type = settings_data["z_max_type"]
            self.x_min_temperature_type = settings_data["x_min_temperature_type"]
            self.x_min_temperature_value = settings_data["x_min_temperature_value"]
            self.x_max_temperature_type = settings_data["x_max_temperature_type"]
            self.x_max_temperature_value = settings_data["x_max_temperature_value"]
            self.y_min_temperature_type = settings_data["y_min_temperature_type"]
            self.y_min_temperature_value = settings_data["y_min_temperature_value"]
            self.y_max_temperature_type = settings_data["y_max_temperature_type"]
            self.y_max_temperature_value = settings_data["y_max_temperature_value"]
            self.z_min_temperature_type = settings_data["z_min_temperature_type"]
            self.z_min_temperature_value = settings_data["z_min_temperature_value"]
            self.z_max_temperature_type = settings_data["z_max_temperature_type"]
            self.z_max_temperature_value = settings_data["z_max_temperature_value"]
            
            self.iter_step_count = settings_data["iter_step_count"]
            self.solve_equation_step_count = settings_data["solve_equation_step_count"]
            self.relax_factor = settings_data["relax_factor"]
            self.solve_equation_tolerance = settings_data["solve_equation_tolerance"]
            self.residual_error = settings_data["residual_error"]
            
            self.save_residual_frequency = settings_data["save_residual_frequency"]
            self.output_frequency = settings_data["output_frequency"]
            self.output_folder = settings_data["output_folder"]
            self.linear_equation_residual_filename = settings_data["linear_equation_residual_filename"]
            self.nonlinear_equation_residual_filename = settings_data["nonlinear_equation_residual_filename"]
            self.pressure_vtk_data_filename = settings_data["pressure_vtk_data_filename"]
            self.temperature_vtk_data_filename = settings_data["temperature_vtk_data_filename"]
            self.velocity_vtk_data_filename = settings_data["velocity_vtk_data_filename"]
            self.dat_filename = settings_data["dat_filename"]
            
            self.is_test = settings_data["is_test"]
            self.is_show_figure = settings_data["is_show_figure"]
            

    def get_mesh_settings(self):
        '''获取网格相关设置'''
        return self.dim,self.x_cell,self.y_cell,self.z_cell,self.coordinate_limit_count
    
    def get_fluid_settingsa(self):
        '''获取流体属性相关设置'''
        return self.density,self.mu,self.specific_heat_capacity,self.conductivity_coefficient
    
    def get_case_settings(self):
        '''获取网格数据相关设置'''
        return self.init_temperature,self.u
    
    def get_boundary_settings(self):
        '''获取边界条件相关设置'''
        return self.x_min_type,self.x_max_type,self.y_min_type,\
                self.y_max_type,self.z_min_type,self.z_max_type,\
                self.x_min_temperature_type,self.x_min_temperature_value,\
                self.x_max_temperature_type,self.x_max_temperature_value,\
                self.y_min_temperature_type,self.y_min_temperature_value,\
                self.y_max_temperature_type,self.y_max_temperature_value,\
                self.z_min_temperature_type,self.z_min_temperature_value,\
                self.z_max_temperature_type,self.z_max_temperature_value
    
    def get_solve_settings(self):
        '''获取求解相关设置'''
        return self.iter_step_count,self.solve_equation_step_count,self.relax_factor,self.solve_equation_tolerance,self.residual_error
    
    def get_post_settings(self):
        '''获取后处理相关设置'''
        return self.save_residual_frequency,self.output_frequency,self.output_folder,self.linear_equation_residual_filename,\
               self.nonlinear_equation_residual_filename,self.pressure_vtk_data_filename,self.temperature_vtk_data_filename,\
               self.velocity_vtk_data_filename,self.dat_filename
    
    def get_other_settings(self):
        '''获取其他设置'''
        return self.is_test,self.is_show_figure