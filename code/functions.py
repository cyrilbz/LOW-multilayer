# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:41:55 2024

@author: cbozonnet
"""

import numpy as np

class parameters():
    def __init__(self):
        
        # activate wall synthesis regulation
        self.wall_regul = True
        
        # individual elastic properties 
        # Obtained from Song 2021 JPhysChem
        Ef = 198e9 # MF Young's modulus
        Em = 10e6 # matrix Young's modulus
        num = 0.3 # matrix Poisson coef
        nuf = 0.22 # MF Poisson coeff
        Gf = 6.06e9 # fiber shear modulus
        Gm = Em/(2*(1+num)) # matrix shear modulus

        # geometric parameters
        self.Vf = 0.1 # MF volume fraction
        self.xi = 1.5 # reinforcement (1: hexagonal array ; 2: square array ; 1.5: random)
        self.AR = 100 # MF aspect ratio

        # Compute parameters from Halpin-Tsai equations
        eta_L = ((Ef/Em)-1)/((Ef/Em)+self.xi*self.AR)
        eta_T = ((Ef/Em)-1)/((Ef/Em)+self.xi)
        eta_G = ((Gf/Gm)-1)/((Gf/Gm)+self.xi)
        eta_nu = ((nuf/num)-1)/((nuf/num)+self.xi)

        # Compute the elastic parameters (Halpin-Tsai equations)
        E1 = Em*(1+self.xi*self.AR*eta_L*self.Vf)/(1-eta_L*self.Vf)
        E2 = Em*(1+self.xi*eta_T*self.Vf)/(1-eta_T*self.Vf)
        G12 = Gm*(1+self.xi*eta_G*self.Vf)/(1-eta_G*self.Vf)
        nu12 = num*(1+self.xi*eta_nu*self.Vf)/(1-eta_nu*self.Vf)
        nu21 = nu12*E2/E1

        # compute elastic matrix coefiicient in plane stress conditions:
        Q11 = E1/(1-nu12*nu21)
        Q22 = E2/(1-nu12*nu21)
        Q12 = nu21*Q11
        Q66 = G12
        
        # Compute the elastic matrix
        self.C_el = np.array(
                    [[Q11,Q12,0],
                    [Q12, Q22, 0],
                    [0,0,Q66]])
        
        # Time constant for viscous flow
        self.tau = 5 * 3600

        # Yield deformation
        self.eps0 = 0.05 
        
        self.E= 100e6
        self.sig_Y = 1e6
        
        # Hydraulic parameters
        self.kh = 1*1e-16 # hydraulic conduct in m/Pa/s [Dumais, new Phytol., 2021]
        self.Psi_src = 0 # (=P-PI)_{ext} : external (xylem) potential in Pa
        self.delta_PsiX = 0 # water source potential amplitude
        self.P_ext = 0 # external pressure      
        self.Pi0 = 1e6 # initial osmotic potential in Pa [Uggla et al, Plant phy, 2001]
        
        # Wall synthesis parameters [Friend et al, Nature com., 2022]
        self.omega = 2.2e-4 # normalised rate of mass growth (kg/m3/s) at T0
        self.omega = 1*self.omega # custom value
        self.Eaw = 1.43 # activation energy for wall building (eV)
        self.kb = 8.617e-5 # Bolztmann's constant (eV/K)
        self.Km = 14.9 # Michaelis constant for wall synthesis (mol/m3)
        self.MMs = 0.342 # molar mass for sucrose (kg/mol)
        self.rho_w = 1500 # cell wall density in kg/m3 
               
        # Temperature
        self.T0 = 273.15 # Reference temperature in K
        self.T = self.T0 + 15 # Actual temperature in K
        
        # Geometry
        self.Lp = 20e-6 # initial periclinal length
        self.La = 10e-6 # initial anticlnal length
        self.Lz = 10e-6 # initial longitudinal length
        self.Wa0 = 1e-6 # initial wall thickness in m
        self.Wp0 = 1e-6 # initial wall thickness in m
        self.MFA0_deg = 5 # initial MFA angle in degrees
        self.MFA0 = self.MFA0_deg*np.pi/180
        
        # Simulation parameters
        self.t0 = 0
        self.t_end = 100*3600 # final time (s)
        self.dt = 0.5*3600 # time step (s)
        
        # Number of layers
        self.nl = 10 # number of layers
        self.tfirstlayer = (self.t_end - self.t0)/self.nl
      
        # Physical constant
        self.Rg = 8.314 # Ideal gas constant
        
        # Sugar transport
        self.eta_s = 2.6e-16 # sugar diffusion constant (m3/s) [Friend et al, Nature com., 2022]
        self.Cs_ext = 200 # sugar source conentration (mol/m3) [Uggla et al, Plant phy, 2001]

        # derived quantities
        VwaT0 =2*self.Wa0*self.Lz*(self.La-2*self.Wp0) # anticl. wall volume
        VwpT0 =2*self.Wp0*self.Lz*self.Lp # pericl. wall volume
        self.Vh0 = self.La*self.Lp*self.Lz - VwaT0 - VwpT0  #init. water volume
        self.ns0 = self.Vh0*self.Pi0/(self.Rg*self.T) # intial sugar content
        self.P0 = 0#self.sig_Y*2*self.Wa0/(self.Lp-2*self.Wa0) # initial turgor pressure in Pa
        self.sig_a0 = self.P0*self.Lp/(2*self.Wa0) # intial wall stress
        
        # compute inital alpha parameter (Lockhart like)
        self.mu = self.tau*E1 # just to get an indicative value of "wall viscosity"
        A0 = 2*self.Lz*(self.Lp + self.La)
        phia0 = A0*self.kh/self.Vh0 # hydraulic conductivity parameter
        phiw = 1/self.mu*(self.Lp-2*self.Wp0)/(2*self.Wp0) # extensibility
        self.alpha = phia0/(phia0+phiw) 
        
def rotation_matrix_loop(angles):
  """
  This function creates a 3x3xn rotation matrix for a multilayer material.

  Args:
      angles: A 1xn numpy array containing rotation angles for each layer.

  Returns:
      A 3x3xn numpy array representing the rotation matrix for each layer.
  """

  rot_func = lambda angle: np.array([[np.cos(angle), -np.sin(angle), 0],
                                       [np.sin(angle), np.cos(angle), 0],
                                       [0, 0, 1]])

  # Create the rotation matrix stack
  rotation_stack = np.stack([rot_func(angle) for angle in angles], axis=2)

  return rotation_stack


def rotation_matrix_vectorized(angles):
  """
  This function creates a nx3x3 rotation matrix for a n-layered material.

  Args:
      angles: A 1xn numpy array containing rotation angles for each layer.

  Returns:
      A nx3x3 numpy array representing the rotation matrix for each layer.
  """
  # Check for valid input
  if not np.ndim(angles) == 1:
    raise ValueError("Angles input must be a 1D numpy array.")

  # Create cos and sin of angles for vectorization
  cos_angles = np.cos(angles)
  sin_angles = np.sin(angles)

  # Define the rotation matrix using vectorization
  rotation_matrix = np.stack([
      np.column_stack((sin_angles**2, cos_angles**2, 2*cos_angles*sin_angles)),
      np.column_stack((cos_angles**2, sin_angles**2, -2*cos_angles*sin_angles)),
      np.column_stack((-cos_angles*sin_angles, cos_angles*sin_angles, sin_angles**2-cos_angles**2))
  ], axis=1)

  return rotation_matrix

        
class data2save: # this creates a structure to save all datas
    def __init__(self, p, t, sol,tdepo,Ldepo):
        self.p = p
        self.t = t
        self.sol = sol
        self.tdepo = tdepo
        self.Ldepo = Ldepo