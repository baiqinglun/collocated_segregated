'''
求解相关的设置
枚举类型:EquationType、ConvectionScheme、PVCouplingScheme
'''
from enum import Enum
from fp import Fp

class EquationType(Enum):
    """
    方程的类型
    """
    # conduction = 0
    # flow = 1
    # conduction_flow = 2
    CONDUCTION = 0
    FLOW = 1
    CONDUCTION_FLOW = 2

class ConvectionScheme(Enum):
    """
    对流项离散格式
    """
    UPWIND = 0
    CD = 1
    POWER_LOW = 2
    SOU = 3

class PVCouplingScheme(Enum):
    """
    压力速度耦合算法
    """
    SIMPLE = 0
    SIMPLEC = 1

class SolveManager:
    """
    求解相关的设置
    """
    def __init__(self):
        '''
        iter_step_count：迭代步数
        solve_equation_step_count：方程迭代步数
        solve_equation_tolerance：求解方程的残差
        residual_error：残差
        solve_equation_count：线性迭代总个数
        '''
        self.solve_equation_count = None
        self.iter_step_count = 100
        self.dt = Fp(1.0)
        self.is_finish = False

        self.equation_type = EquationType.CONDUCTION
        self.solve_equation_step_count = 10
        self.solve_equation_tolerance = Fp(1.e-1)
        self.relax_factor = Fp(0.75)
        self.residual_error = Fp(2.-6)
        self.solve_equation_total_count = 0

        self.convection_scheme = ConvectionScheme.CD
        self.conduction_scheme = ConvectionScheme.CD

        # 二阶范数
        self.l2_curr = Fp(0.0)
        self.l2_max = Fp(-1.e20)
        self.l2_max_t = Fp(-1.e20)
        self.l2_t = Fp(0.0)

    def set_iter_step_count(self,iter_step_count):
        '''
        设置总迭代步数
        '''
        self.iter_step_count = iter_step_count

    def set_dt(self,dt):
        '''
        设置时间步长
        '''
        self.dt = dt

    def set_is_finish(self,is_finish):
        '''
        模拟是否完成
        '''
        self.is_finish = is_finish

    def set_solve_equation_step_count(self,solve_equation_step_count):
        '''
        设置方程迭代步数
        '''
        self.solve_equation_step_count = solve_equation_step_count

    def set_solve_equation_tolerance(self,solve_equation_tolerance):
        '''
        设置方程求解容差
        '''
        self.solve_equation_tolerance = solve_equation_tolerance

    def set_relax_factor(self, relax_factor):
        '''
        设置松弛因子
        '''
        self.relax_factor = relax_factor

    def set_residual_error(self, residual_error):
        '''
        设置求解残差
        '''
        self.residual_error = residual_error

    def set_solve_equation_count(self, solve_equation_count):
        '''
        设置方程数量
        '''
        self.solve_equation_count = solve_equation_count

    def set_convection_scheme(self,convection_scheme):
        '''
        设置对流项求解格式
        '''
        self.convection_scheme = convection_scheme

    def set_conduction_scheme(self,conduction_scheme):
        '''
        设置扩散项求解格式
        '''
        self.conduction_scheme = conduction_scheme

    def set_equation_type(self,equation_type):
        '''
        设置方程类型
        '''
        self.equation_type = equation_type
