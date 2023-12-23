'''流体相关属性'''
import numpy as np
from mesh import MeshManager
from fp import Fp

class Fluid:
    """
    @name：Fluid
    @description: 定义流体的相关信息
    @variable:
        density:密度
        mu:粘度
        conductivity_coefficient：热传导系数
        specific_heat_capacity：比热
    @function: 
        set_physical_property()：设置流体信息
    """
    def __init__(self,mesh:MeshManager):
        self.n_x_cell = mesh.n_x_cell
        self.n_y_cell = mesh.n_y_cell
        self.n_z_cell = mesh.n_z_cell

        self.density = np.ones((self.n_x_cell,self.n_y_cell,self.n_z_cell),dtype=Fp)
        self.mu = np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.conductivity_coefficient = np.zeros((self.n_x_cell,self.n_y_cell,self.n_z_cell),dtype=Fp)
        self.specific_heat_capacity = Fp(0.0)

        self.source_term = Fp(0.0)

    def set_density(self,density):
        '''设置密度'''
        self.density = density * np.ones((self.n_x_cell,self.n_y_cell,self.n_z_cell),dtype=Fp)

    def set_mu(self,mu):
        '''设置粘度'''
        self.mu = mu * np.ones((self.n_x_cell,self.n_y_cell,self.n_z_cell),dtype=Fp)

    def set_specific_heat_capacity(self,specific_heat_capacity):
        '''设置比热容'''
        self.specific_heat_capacity = specific_heat_capacity

    def set_conductivity_coefficient(self,conductivity_coefficient):
        '''设置热传导系数'''
        self.conductivity_coefficient = conductivity_coefficient * np.ones((self.n_x_cell,self.n_y_cell,self.n_z_cell),dtype=Fp)

    def set_physical_property(self,density,mu,specific_heat_capacity,conductivity_coefficient):
        '''设置所有属性'''
        self.set_density(density)
        self.set_mu(mu)
        self.set_specific_heat_capacity(specific_heat_capacity)
        self.set_conductivity_coefficient(conductivity_coefficient)
