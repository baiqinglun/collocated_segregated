from mesh import MeshManager
import numpy as np
from fp import Fp

class Fluid:
    def __init__(self,mesh:MeshManager):
        self.n_x_cell = mesh.n_x_cell
        self.n_y_cell = mesh.n_y_cell
        self.n_z_cell = mesh.n_z_cell

        self.density = np.ones((self.n_x_cell,self.n_y_cell,self.n_z_cell),dtype=Fp)
        self.mu = np.ones((self.n_x_cell, self.n_y_cell, self.n_z_cell), dtype=Fp)
        self.conductivity_coefficient = Fp(0.0)
        self.specific_heat_capacity = Fp(0.0)

        self.source_term = Fp(0.0)

    def set_density(self,density):
        self.density = density * np.ones((self.n_x_cell,self.n_y_cell,self.n_z_cell),dtype=Fp)

    def set_mu(self,mu):
        self.mu = mu * np.ones((self.n_x_cell,self.n_y_cell,self.n_z_cell),dtype=Fp)

    def set_specific_heat_capacity(self,specific_heat_capacity):
        self.specific_heat_capacity = specific_heat_capacity

    def set_conductivity_coefficient(self,conductivity_coefficient):
        self.conductivity_coefficient = conductivity_coefficient

    def set_physical_property(self,density,mu,specific_heat_capacity,conductivity_coefficient):
        self.set_density(density)
        self.set_mu(mu)
        self.set_specific_heat_capacity(specific_heat_capacity)
        self.set_conductivity_coefficient(conductivity_coefficient)