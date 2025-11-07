# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:41:55 2024

@author: cbozonnet
"""

import numpy as np
import re

class parameters:
    def __init__(self):
        
        # activate wall synthesis regulation
        self.wall_regul = True
        
        # force no MF rotation (if True : no rotation of the MF)
        self.no_rotation = False
        
        # force a constant elongation rate
        self.force_elongation = True
        self.G_forced = 1e-6 # 3e-7for meca prop computations in single layer
        
        # change deposition angle 
        #(if True: attempt to align deposition towards stress eigenvector
        # or progressive linear shift in deposition angle)
        # NB : initialization must probably be improved to use this feature
        self.change_deposition = False
        
        # force pressure steps 
        self.pressure_steps = False
        self.p_init = 0.1e6
        self.deltaP = 0.1e6 # pressure steps
        self.dt_plateau = 50*3600 # length of a plateau 
        
        # individual elastic properties 
        # Obtained from Song 2021 JPhysChem
        self.Ef = 198e9 # MF Young's modulus
        self.Em = 10e6 # matrix Young's modulus
        self.num = 0.3 # matrix Poisson coef
        self.nuf = 0.22 # MF Poisson coeff
        self.Gf = 6.06e9 # fiber shear modulus
        self.Gm = self.Em/(2*(1+self.num)) # matrix shear modulus

        # geometric parameters
        self.Vf = 0.1 # MF volume fraction
        self.xi = 1.5 # reinforcement (1: hexagonal array ; 2: square array ; 1.5: random)
        self.AR = 100 # MF aspect ratio
        
        # Time constant for viscous flow
        self.tau_visc = 5 * 3600
        
        # time constant for microtubules reorientation
        self.tau_MT = 6*3600 

        # Yield deformation
        self.eps0 = 0.03
        
        # MFA angle
        self.MFA0_deg = 5 # initial MFA angle in degrees
        self.MFA0 = self.MFA0_deg*np.pi/180
        
        # final Micro-tubule angle
        self.theta_MT_max = 45*np.pi/180
        
        # Hydraulic parameters 
        # to compute water fluxes when elongation rate is not imposed
        self.kh = 1*1e-16 # hydraulic conduct in m/Pa/s [Dumais, new Phytol., 2021]
        self.Psi_src = 0 # (=P-PI)_{ext} : external (xylem) potential in Pa
        self.delta_PsiX = 0 # water source potential amplitude
        self.P_ext = 0 # external pressure  
        self.Pi0 = 0.2e6 # initial osmotic potential in Pa [Uggla et al, Plant phy, 2001]
        
        # Wall synthesis parameters [Friend et al, Nature com., 2022]
        # in case wall synthesis is not regulated
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
    
        # Simulation parameters
        self.t0 = 0
        self.t_end = 500*3600 # final time (s)
        self.dt = 0.1*3600 # time step at which data is saved (s) (not the actual time step)
        
        # Number of layers
        self.nl = 80 # number of layers
        self.tfirstlayer = (self.t_end - self.t0)/self.nl
      
        # Physical constant
        self.Rg = 8.314 # Ideal gas constant
        
        # Sugar transport 
        # in case there is no osmoregulation
        self.eta_s = 2.6e-16 # sugar diffusion constant (m3/s) [Friend et al, Nature com., 2022]
        self.Cs_ext = 200 # sugar source conentration (mol/m3) [Uggla et al, Plant phy, 2001]
        
        # some initial variables
        VwaT0 =2*self.Wa0*self.Lz*(self.La-2*self.Wp0) # anticl. wall volume
        VwpT0 =2*self.Wp0*self.Lz*self.Lp # pericl. wall volume
        self.Vh0 = self.La*self.Lp*self.Lz - VwaT0 - VwpT0  #init. water volume
        self.ns0 = self.Vh0*self.Pi0/(self.Rg*self.T) # intial sugar content
        
        # initialize some parameters that are computed later
        self.C_el = None
        self.Y1 = None
        self.Y2 = None
        self.Y12 = None
        self.angle = None
        self.radii = None
             
        # compute the previous parameters
        self.compute_dependencies()
        
###################### Here below compute different dependencies
    def compute_dependencies(self):

        # Compute parameters from Halpin-Tsai equations
        eta_L = ((self.Ef/self.Em)-1)/((self.Ef/self.Em)+self.xi*self.AR)
        eta_T = ((self.Ef/self.Em)-1)/((self.Ef/self.Em)+self.xi)
        eta_G = ((self.Gf/self.Gm)-1)/((self.Gf/self.Gm)+self.xi)
        eta_nu = ((self.nuf/self.num)-1)/((self.nuf/self.num)+self.xi)

        # Compute the elastic parameters (Halpin-Tsai equations)
        E1 = self.Em*(1+self.xi*self.AR*eta_L*self.Vf)/(1-eta_L*self.Vf)
        E2 = self.Em*(1+self.xi*eta_T*self.Vf)/(1-eta_T*self.Vf)
        G12 = self.Gm*(1+self.xi*eta_G*self.Vf)/(1-eta_G*self.Vf)
        nu12 = self.num*(1+self.xi*eta_nu*self.Vf)/(1-eta_nu*self.Vf)
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
        
        # Tsai-hill criterion parameters
        self.Y1 = Q11*self.eps0
        self.Y2 = Q22*self.eps0
        self.Y12 = Q66*self.eps0
        
        # perform change of variable for ellipsoid projection
        self.angle= 0.5*np.arctan(self.Y2**2/(self.Y2**2-self.Y1**2)) # angle of rotation
        # here below compute the radii
        a_1 = np.sqrt(1/(np.cos(self.angle)**2/self.Y1**2 + np.sin(self.angle)**2/self.Y2**2 + np.cos(self.angle)*np.sin(self.angle)/self.Y1**2))
        a_2 = np.sqrt(1/(np.sin(self.angle)**2/self.Y1**2 + np.cos(self.angle)**2/self.Y2**2 -np.cos(self.angle)*np.sin(self.angle)/self.Y1**2))
        a_3 = self.Y12
        self.radii = np.array((a_1,a_2,a_3)) # semi axis of the ellipsoid
        
    def set_param(self, param_name, new_value):
        """
        Modifies a parameter and recomputes dependent values.

        Args:
            param_name (str): The name of the parameter to modify.
            new_value: The new value for the parameter.
        """

        if hasattr(self, param_name):
            setattr(self, param_name, new_value)
            self.compute_dependencies()  # Recompute dependencies after modification
        else:
            raise Exception("Parameter not known !")
        
# def rotation_matrix_loop(angles):
#   """
#   This function creates a 3x3xn rotation matrix for a multilayer material.

#   Args:
#       angles: A 1xn numpy array containing rotation angles for each layer.

#   Returns:
#       A 3x3xn numpy array representing the rotation matrix for each layer.
#   """

#   rot_func = lambda angle: np.array([[np.cos(angle), -np.sin(angle), 0],
#                                        [np.sin(angle), np.cos(angle), 0],
#                                        [0, 0, 1]])

#   # Create the rotation matrix stack
#   rotation_stack = np.stack([rot_func(angle) for angle in angles], axis=2)

#   return rotation_stack


def rotation_matrix_stress(angles):
  """
  This function creates a nx3x3 STRESS rotation matrix for a n-layered material.

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
      np.column_stack((cos_angles**2, sin_angles**2, 2*cos_angles*sin_angles)),
      np.column_stack((sin_angles**2, cos_angles**2, -2*cos_angles*sin_angles)),
      np.column_stack((-cos_angles*sin_angles, cos_angles*sin_angles, cos_angles**2-sin_angles**2))
  ], axis=1)

  return rotation_matrix

def rotation_matrix_strain(angles):
  """
  This function creates a nx3x3 STRAIN rotation matrix for a n-layered material.

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
      np.column_stack((cos_angles**2, sin_angles**2, cos_angles*sin_angles)),
      np.column_stack((sin_angles**2, cos_angles**2, -cos_angles*sin_angles)),
      np.column_stack((-2*cos_angles*sin_angles, 2*cos_angles*sin_angles, cos_angles**2-sin_angles**2))
  ], axis=1)

  return rotation_matrix

def compute_thresholds(coefs, angle, sa, sl, tau, plasticity): 
    """
    This function computes the yield thresholds according to Tsai-Hill criterion
    Args:
        coefs: a 3x1 array containing the (ideal) ellispoid semi-axes
        angle: a scalar, the angle used for the change of variable
        sa: a nx1 vector containing the anticlinal stress values
        sl: a nx1 vector containing the longitudinal stress values
        tau: a nx1 vector containing the shear stress values
        plasticity: a nx1 Boolean vector containing the Tsai-Hill criterion result

    Returns:
        A nx3 numpy array containing the yield thresholds
    """
    if isinstance(sa, (np.ndarray, list, tuple)): # check the nature of the input
        n_layers = len(sa) # the number of layers
    else:
        n_layers = 1  # Return 1 for non-array sa
    
    thresholds = np.zeros((n_layers,3)) 
    
    for index in range(n_layers):      
        if not plasticity[index]:  # assign 0 where there is no plasticity
            thresholds[index,:] = 0 
        else:
            #perform change of variable
            y1 = np.cos(angle)*sa[index] - np.sin(angle)*sl[index]
            y2 = np.sin(angle)*sa[index] + np.cos(angle)*sl[index]
            point = [y1,y2,tau[index]]      
            
            # apply constrained optimization problem
            lam = 0 # inital value for Lagrange multiplier
            n=0 # number of Newton iteration
            n_max = 60 # max number of iteration
            res = 1e-6 # target residual
            glam = 10*res # intial value to enter the loop
            while abs(glam)>res and n<n_max: # Newton algorithm to find lambda
                # compute g(lambda) function (objective function to nullify)
                glam = (point[0]/(coefs[0]*(1-lam/coefs[0]**2)))**2 \
                    + (point[1]/(coefs[1]*(1-lam/coefs[1]**2)))**2 \
                    + (point[2]/(coefs[2]*(1-lam/coefs[2]**2)))**2 -1  
                # compute dg/dlmabda 
                dglam = 2*point[0]**2/(coefs[0]**4*(1-lam/coefs[0]**2)**3) \
                    + 2*point[1]**2/(coefs[1]**4*(1-lam/coefs[1]**2)**3) \
                    + 2*point[2]**2/(coefs[2]**4*(1-lam/coefs[2]**2)**3)             
                lam = lam - glam/dglam # Newton iteration
                n=n+1
            if (n>=n_max): 
                raise Exception("Projection onto the ellipsoid did not work")
            # compute solution
            x1 = point[0]/(1-lam/coefs[0]**2)
            x2 = point[1]/(1-lam/coefs[1]**2)
            x3 = point[2]/(1-lam/coefs[2]**2)
            
            # perform change of variable
            thresholds[index,0] = np.cos(angle)*x1 + np.sin(angle)*x2
            thresholds[index,1] = - np.sin(angle)*x1 + np.cos(angle)*x2
            thresholds[index,2] = x3
    return thresholds

        
class data2save: # this creates a structure to save all datas
    def __init__(self, p, t, sol,tdepo,Ldepo,theta_depo):
        self.p = p
        self.t = t
        self.sol = sol
        self.tdepo = tdepo
        self.Ldepo = Ldepo
        self.theta_depo = theta_depo
        
class data2write: # a structure to write some post processed data
    def __init__(self, name, parameter, sY, E, epsY, tau_visc):
        self.name = name
        self.values = parameter
        self.sY= sY
        self.E = E
        self.epsY = epsY
        self.tau_visc = tau_visc
        
class data2write_multi: # a structure to write some post processed data
    def __init__(self, name, parameter, stress):
        self.name = name
        self.values = parameter
        self.stress = stress

        
def sort_files(files,pattern):
  """
  This function sorts the list of pressure files by the numerical value of 'XX' in the filename.

  Args:
      pressure_files (list): A list of pressure file names.

  Returns:
      list: The sorted list of pressure file names.
  """
  # Define a key function to extract and sort by the numerical value of 'XX'
  def key(filename):
    match = re.match(pattern, filename)
    if match:
      return int(match.group(1))  # Extract and convert 'XX' to integer
    else:
      return float('inf')  # Place files without the pattern at the end

  # Sort the list using the key function
  sorted_files = sorted(files, key=key)
  return sorted_files