# collocated_segregated
分离求解器

# 文档使用

## 对流扩散方程验证

修改配置文件
```json
{
  "dim": 2,
  "x_cell": 3,
  "y_cell": 1,
  "z_cell": 1,
  "coordinate_limit_count":[-0.25, 1.25, 0, 1, 0, 1],
  "init_temperature": 0.5,
  "density": 1.0,
  "mu": 1.0,
  "specific_heat_capacity": 0.0,
  "conductivity_coefficient": 1000000,
  "x_min_type" : "wall",
  "x_max_type" : "wall",
  "y_min_type" : "wall",
  "y_max_type" : "wall",
  "z_min_type" : "wall",
  "z_max_type" : "wall",
  "x_min_temperature_type": "constant",
  "x_min_temperature_value": 1000,
  "x_max_temperature_type": "constant",
  "x_max_temperature_value": 373,
  "y_min_temperature_type": "constant",
  "y_min_temperature_value": 373,
  "y_max_temperature_type": "constant",
  "y_max_temperature_value": 293,
  "z_min_temperature_type": "constant",
  "z_min_temperature_value": 375,
  "z_max_temperature_type": "constant",
  "z_max_temperature_value": 375,
  "iter_step_count": 100,
  "solve_equation_step_count": 20,
  "relax_factor": 0.75,
  "solve_equation_tolerance": 0.01,
  "residual_error": 1e-6,
  "save_residual_frequency": 20,
  "output_frequency": 20,
  "output_folder": "output",
  "linear_equation_residual_filename" : "linear_equation_residual.res",
  "nonlinear_equation_residual_filename" : "nonlinear_equation_residual.res",
  "vtk_data_filename" : "data.vtk",
  "dat_filename" : "temperature_line.dat",
  "is_test": false,
  "is_show_figure": false,
  "u": 1.0

}
```
将solve里的求解方程改为
```python
EquationType.conduction_flow
```

## 扩散方程测试

主文件
```python
solveManager.set_convection_scheme(ConvectionScheme.cd)
solveManager.set_conduction_scheme(ConvectionScheme.cd)
solveManager.set_equation_type(EquationType.conduction)
```

```json
{
  "dim": 2,
  "x_cell": 20,
  "y_cell": 20,
  "z_cell": 1,
  "coordinate_limit_count":[0, 10, 0, 10, 0, 1],
  "init_temperature": 373,
  "density": 1.0,
  "mu": 0.001,
  "specific_heat_capacity": 1.0,
  "conductivity_coefficient": 81.0,
  "x_min_type" : "wall",
  "x_max_type" : "wall",
  "y_min_type" : "wall",
  "y_max_type" : "wall",
  "z_min_type" : "wall",
  "z_max_type" : "wall",
  "x_min_temperature_type": "constant",
  "x_min_temperature_value": 1000,
  "x_max_temperature_type": "constant",
  "x_max_temperature_value": 373,
  "y_min_temperature_type": "constant",
  "y_min_temperature_value": 373,
  "y_max_temperature_type": "constant",
  "y_max_temperature_value": 293,
  "z_min_temperature_type": "constant",
  "z_min_temperature_value": 375,
  "z_max_temperature_type": "constant",
  "z_max_temperature_value": 375,
  "iter_step_count": 100,
  "solve_equation_step_count": 20,
  "relax_factor": 0.75,
  "solve_equation_tolerance": 0.01,
  "residual_error": 1e-6,
  "save_residual_frequency": 20,
  "output_frequency": 20,
  "output_folder": "output",
  "linear_equation_residual_filename" : "linear_equation_residual.res",
  "nonlinear_equation_residual_filename" : "nonlinear_equation_residual.res",
  "vtk_data_filename" : "data.vtk",
  "dat_filename" : "temperature_line.dat",
  "is_test": false,
  "is_show_figure": false,
  "u": 0.0
}
```