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
        
        # Mechanical parameters
        self.E = 100e6 # Young Modulus - it is 30e6 in [Dumais, new Phytol., 2021]
        self.mu = 0.8e10 # wall viscosity in Pa.s [Dumais, new Phytol., 2021]
        tau = 10 * 3600
        #self.mu = 500*self.mu # custom value
        self.mu = self.E*tau
        self.sig_Y = 5e6 # Plastic threshold in Pa [user choice]
        
        # Hydraulic parameters
        self.kh = 1000*1e-16 # hydraulic conduct in m/Pa/s [Dumais, new Phytol., 2021]
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
        self.t_end = 200*3600 # final time (s)
        self.dt = 0.5*3600 # time step (s)
        
        # Number of layers
        self.nl = 40 # number of layers
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
        A0 = 2*self.Lz*(self.Lp + self.La)
        phia0 = A0*self.kh/self.Vh0 # hydraulic conductivity parameter
        phiw = 1/self.mu*(self.Lp-2*self.Wp0)/(2*self.Wp0) # extensibility
        self.alpha = phia0/(phia0+phiw) 
        
class data2save: # this creates a structure to save all datas
    def __init__(self, p, t, sol,tdepo,Ldepo):
        self.p = p
        self.t = t
        self.sol = sol
        self.tdepo = tdepo
        self.Ldepo = Ldepo