from fp import Fp
from enum import Enum
from post_process import PostProcessManager

class EquationType(Enum):
    conduction = 0
    flow = 1
    conduction_flow = 2

class ConvectionScheme(Enum):
    upwind = 0
    cd = 1
    power_law = 2
    sou = 3

class SolveManager:
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

        self.equation_type = EquationType.conduction
        self.solve_equation_step_count = 10
        self.solve_equation_tolerance = Fp(1.e-1)
        self.relax_factor = Fp(0.75)
        self.residual_error = Fp(2.-6)
        self.solve_equation_total_count = 0

        self.convection_scheme = ConvectionScheme.upwind

        # 二阶范数
        self.l2_curr = Fp(0.0)
        self.l2_max = Fp(-1.e20)
        self.l2_max_t = Fp(-1.e20)
        self.l2_t = Fp(0.0)

    def set_iter_step_count(self,iter_step_count):
        self.iter_step_count = iter_step_count

    def set_dt(self,dt):
        self.dt = dt

    def set_is_finish(self,is_finish):
        self.is_finish = is_finish

    def set_solve_equation_step_count(self,solve_equation_step_count):
        self.solve_equation_step_count = solve_equation_step_count

    def set_solve_equation_tolerance(self,solve_equation_tolerance):
        self.solve_equation_tolerance = solve_equation_tolerance

    def set_relax_factor(self, relax_factor):
        self.relax_factor = relax_factor

    def set_residual_error(self, residual_error):
        self.residual_error = residual_error

    def set_solve_equation_count(self, solve_equation_count):
        self.solve_equation_count = solve_equation_count

    def set_convection_scheme(self,convection_scheme):
        self.convection_scheme = convection_scheme

    def iter_step_count(self):
        return self.iter_step_count

    def dt(self):
        return self.dt

    def is_finish(self):
        return self.is_finish

    def solve_equation_step_count(self):
        return self.solve_equation_step_count

    def solve_equation_tolerance(self):
        return self.solve_equation_tolerance

    def relax_factor(self,):
        return self.relax_factor

    def residual_error(self,):
        return self.residual_error

    def solve_equation_count(self,):
        return self.solve_equation_count

    def convection_scheme(self):
        return self.convection_scheme

    def solve(self,postProcessManager:PostProcessManager):
        for iter_step in range(1,self.iter_step_count+1):
            if iter_step == 1 or iter_step % postProcessManager.save_residual_frequency == 0 or iter_step == self.iter_step_count:
                print('')
                print('---------------------------')
                print('Begin iter = ', iter_step)
                print('---------------------------')

        # self.is_finish() =