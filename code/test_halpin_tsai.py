# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:08:11 2024

@author: cbozonnet
"""

# Test program for Halpin-Tsai equations

# list all parmeters [SI units]

# individual elastic properties
Ef = 198e9 # MF Young's modulus
Em = 10e6 # matrix Young's modulus
num = 0.3 # matrix Poisson coef
nuf = 0.22 # MF Poisson coeff
Gf = 6.06e9 # fiber shear modulus
Gm = Em/(2*(1+num)) # matrix shear modulus

# geometric parameters
Vf = 0.05 # MF volume fraction
xi = 1.5 # reinforcement (1: hexagonal array ; 2: square array ; 1.5: random)
AR = 100 # MF aspect ratio


# Compute parameters from Halpin-Tsai equations
eta_L = ((Ef/Em)-1)/((Ef/Em)+xi*AR)
eta_T = ((Ef/Em)-1)/((Ef/Em)+xi)
eta_G = ((Gf/Gm)-1)/((Gf/Gm)+xi)
eta_nu = ((nuf/num)-1)/((nuf/num)+xi)

# Compute the elastic parameters
E1 = Em*(1+xi*AR*eta_L*Vf)/(1-eta_L*Vf)/1e6
E2 = Em*(1+xi*eta_T*Vf)/(1-eta_T*Vf)/1e6
G12 = Gm*(1+xi*eta_G*Vf)/(1-eta_G*Vf)/1e6
nu12 = num*(1+xi*eta_nu*Vf)/(1-eta_nu*Vf)
nu21 = nu12*E2/E1

# compute elastic matrix coefiicient:
Q11 = E1/(1-nu12*nu21)
Q22 = E2/(1-nu12*nu21)
Q12 = nu21*Q11
Q66 = G12