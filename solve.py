'''
求解相关的设置
枚举类型:
    - 方程类型EquationType
    - 离散格式DiscreteScheme
    - 压力速度耦合算法PVCouplingScheme
'''
from enum import Enum
from fp import Fp

class EquationType(Enum):
    """方程的类型"""
    DIFFUSION = 0
    CONVECTION = 1
    DIFFUSION_CONVECTION = 2

class DiscreteScheme(Enum):
    """离散格式"""
    UPWIND = 0
    CD = 1
    POWER_LOW = 2
    SOU = 3

class PVCouplingScheme(Enum):
    """压力速度耦合算法"""
    SIMPLE = 0
    SIMPLEC = 1

class SolveManager:
    """求解相关的设置"""
    def __init__(self):
        self.solve_equation_count:int = None
        self.dt = Fp(1.0)
        self.is_finish = False

        self.equation_type = EquationType.DIFFUSION_CONVECTION
        self.solve_equation_step_count = 10
        self.solve_equation_tolerance = Fp(1.e-1)
        self.solve_equation_total_count = 0

        # 离散格式
        self.diffusion_scheme = DiscreteScheme.CD # 对流
        self.conduction_scheme = DiscreteScheme.CD # 扩散

        # 迭代步数
        self.iter_step_count  = 100
        self.iter_step_count_u = 100
        self.iter_step_count_v = 100
        self.iter_step_count_w = 100
        self.iter_step_count_p = 100
        self.iter_step_count_t = 10

        # 松弛因子
        self.relax_factor_u = Fp(0.0)
        self.relax_factor_v = Fp(0.0)
        self.relax_factor_w = Fp(0.0)
        self.relax_factor_p = Fp(0.0)
        self.relax_factor_t = Fp(0.75)

        # 残差
        self.residual_error_u = Fp(1.e-2)
        self.residual_error_v = Fp(1.e-2)
        self.residual_error_w = Fp(1.e-2)
        self.residual_error_p = Fp(1.e-2)
        self.residual_error_t = Fp(1.e-2)

        self.mass_total = Fp(1.e-6)
        self.temperature_total = Fp(1.e-6)

        # 二阶范数
        self.l2_curr = Fp(0.0)
        self.l2_max = Fp(-1.e20)
        self.l2_max_u = Fp(-1.e20)
        self.l2_max_v = Fp(-1.e20)
        self.l2_max_w = Fp(-1.e20)
        self.l2_max_p = Fp(-1.e20)
        self.l2_max_pp = Fp(-1.e20)
        self.l2_max_t = Fp(-1.e20)
        self.l2_u = Fp(0.0)
        self.l2_v = Fp(0.0)
        self.l2_w = Fp(0.0)
        self.l2_p = Fp(0.0)
        self.l2_pp = Fp(0.0)
        self.l2_t = Fp(0.0)

    def set_dt(self,dt):
        '''设置时间步长'''
        self.dt = dt

    def set_relax_factor(self,relax_factor):
        '''设置松弛因子'''
        self.relax_factor_p = relax_factor
        self.relax_factor_t = relax_factor
        self.relax_factor_u = relax_factor
        self.relax_factor_v = relax_factor
        self.relax_factor_w = relax_factor

    def set_is_finish(self,is_finish):
        '''模拟是否完成'''
        self.is_finish = is_finish

    def set_iter_step_count(self,iter_step_count):
        '''设置总迭代步数'''
        self.iter_step_count = iter_step_count

    def set_solve_equation_step_count(self,solve_equation_step_count):
        '''设置方程迭代步数'''
        self.solve_equation_step_count = solve_equation_step_count

    def set_solve_equation_tolerance(self,solve_equation_tolerance):
        '''设置方程求解容差'''
        self.solve_equation_tolerance = solve_equation_tolerance

    def set_residual_error(self,residual_error):
        '''设置残差'''
        self.residual_error_p = residual_error
        self.residual_error_t = residual_error
        self.residual_error_u = residual_error
        self.residual_error_v = residual_error
        self.residual_error_w = residual_error

    def set_solve_equation_count(self, solve_equation_count):
        '''设置方程数量'''
        self.solve_equation_count = solve_equation_count

    def set_diffusion_scheme(self,convection_scheme):
        '''设置对流项求解格式'''
        self.diffusion_scheme = convection_scheme

    def set_conduction_scheme(self,conduction_scheme):
        '''设置扩散项求解格式'''
        self.conduction_scheme = conduction_scheme

    def set_equation_type(self,equation_type):
        '''设置方程类型'''
        self.equation_type = equation_type
