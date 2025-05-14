# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:53:09 2024

@author: cbozonnet
"""
import numpy as np
from scipy.integrate import odeint, solve_ivp
import pickle
from functions import data2save, parameters, rotation_matrix_strain, rotation_matrix_stress, compute_thresholds
import time
import os
import sys

def dydt(t,y,p): # all physical equations for time integration
    # NB : to switch from odeint to solve_ivp and vice versa
    # you must switch t and y in the previous line
    # and comment/uncomment the call to the solver below
    
    # get global variables - useful to handle the multilayer approach
    global iteration_count, tdepo, Ldepo, t_pressure, k_pressure
    
    # extract data from y vector
    ns = y[0] # extract sugar content
    La = y[1] # extract cell length
    sa = y[2:2+p.nl] # extract anticlinal stress
    sl = y[2+p.nl:2+2*p.nl] # extract longitudinal stress
    tau = y[2+2*p.nl:2+3*p.nl] # extract longitudinal stress
    Wa = y[2+3*p.nl:2+4*p.nl+1] # extract anticlinal wall thickness

    # Initialize time derivatives
    dnsdt = 0
    dLdt = 0
    dsadt = np.zeros((p.nl,1))
    dsldt = np.zeros((p.nl,1)) 
    dtaudt = np.zeros((p.nl,1)) 
    dWadt = np.zeros((p.nl,1))
    
    # Initialize some other vectors
    eps_t = np.zeros((p.nl)) # total deformation
    MFA = np.zeros((p.nl)) # micro fibril angles
    
    # Go to next layer
    # if time is sufficiently high and we did not yet reach the max number of layers (>1)
    if (tdepo[iteration_count]<t and iteration_count < len(tdepo)-1 and p.nl>1):
        iteration_count += 1 # switch to next layer
        Ldepo[iteration_count] = La # save deposition length
        
    # handle pressure forcing
    p_forcing = 0
    if (p.pressure_steps==True):
        if (k_pressure < len(t_pressure)-1): 
            if (t_pressure[k_pressure+1]<t): 
                k_pressure +=1 # update increment      
        p_forcing = p.p_init + k_pressure*p.deltaP # compute pressure forcing
 
    # Compute volumes
    WaT = sum(Wa)
    WpT = p.Wp0
    VwaT = 2*WaT*p.Lz*(La-2*WpT)
    VwpT = 2*WpT*p.Lp*p.Lz
    Vh = La*p.Lp*p.Lz - VwpT- VwaT
    
    # Compute sugar and wall synthesis related quantities
    Cs = ns/Vh # Sugar concentration (mol/m3)
    PI =  Cs*p.Rg*p.T # Cell osmotic potential (Pa)
    fs = p.eta_s*(p.Cs_ext - Cs) # sugar flux
    dMmax = p.omega*Vh*np.exp(p.Eaw/p.kb*(1/p.T0-1/p.T)) # Maximal speed of mass increment 
    dMdt = dMmax*Cs/(Cs+p.Km) # Mass growth (kg/s)
    dVwadt = 1/p.rho_w*dMdt # wall synthesis in volume  
    
    # Compute mean stresses and pressure
    sa_mean = np.dot(sa,Wa)/WaT # mean anticlinal stress
    P = 2*sa_mean*WaT/(p.Lp-2*WaT) + p.P_ext # pressure
    
    # Compute the total deformation and MFA related quantitites
    eps_t[:(iteration_count+1)] = (La - Ldepo[:(iteration_count+1)])/Ldepo[:(iteration_count+1)]    
    if (p.no_rotation==True): 
        MFA = np.arctan(np.tan(p.MFA0)*np.exp(np.zeros((p.nl)))) # force no rotation
    else: 
        MFA = np.arctan(np.tan(p.MFA0)*np.exp(eps_t)) # MFA evolution
        
    # Compute water fluxes
    my_psiX = p.Psi_src + 0.5*p.delta_PsiX*(1+np.cos((t/3600+12-0)*np.pi/12))
    A = 2*p.Lz*(p.Lp+La) # area for water fluxes
    Q = A*p.kh*(my_psiX-P+PI+p_forcing) # water fluxes
    
    # compute the anticlinal elongation rate (ERa)
    ERa = (1-2*WaT/p.Lp)*(Q+dVwadt)/Vh
    if (p.wall_regul==True): ERa = Q/Vh # wall thickness regulation
    # ERa = 1e-5

    # Compute the anticlinal length changes
    dLdt=(La-2*WpT)*ERa
    
    # Wall synthesis when thickness regulation
    if (p.wall_regul==True): dVwadt=2*p.Lz*WaT*dLdt
    
    # Compute the changes in wall layer thicknesses
    dWadt = -Wa*1/(La-2*WpT)*(dLdt) # thickness evolution for all layers
    dWadt[iteration_count] += dVwadt/(2*p.Lz*(La-2*WpT)) # wall synthesis in a target layer
    
    # compute changesin sugar content 
    #dnsdt = -1/p.MMs*dMdt + fs # no osmoregulation
    dnsdt = ns/Vh*(dLdt*(p.Lp*p.Lz-2*WaT*p.Lz)-2*p.Lz*sum(dWadt)*(La-2*WpT)) # omoregulation
    
    # Compute rotation matrix and compute strains in the MF frame
    Rotate = rotation_matrix_strain(np.pi/2*np.ones_like(MFA)-MFA) # a nlx3x3 rotation matrix
    eps_MF = Rotate @ np.array((ERa,0,0)) # a nlx3 matrix containing all deformation rates
    eps_MF_tr = np.transpose(eps_MF) # 3 x nl matrix

    # Compute the stress in the MF frame 
    Rotate_stress = rotation_matrix_stress(np.pi/2*np.ones_like(MFA)-MFA) # a nlx3x3 rotation matrix
    s_rotated = np.zeros((3, len(MFA)))    
    for i in range(len(MFA)):
      # Perform multiplication for each layer -> IT IS NOT VECTORIZED !!! =(
      s_rotated [:, i] = Rotate_stress[i, :, :]  @ np.array((sa[i],sl[i],tau[i]))
    
    # Check Tsai-Hill criterion
    Tsai = (s_rotated[0,:]/p.Y1)**2 - (s_rotated[0,:]*s_rotated[1,:])/(p.Y1**2) + (s_rotated[1,:]/p.Y2)**2 + (s_rotated[2,:]/p.Y12)**2
    plasticity = (Tsai>1) # True if the wall layer is in plasticity
        
    # Compute Yield thresholds ( outputs a nlx3 matrix) and plastic components
    yields = compute_thresholds(p.radii, p.angle, s_rotated[0,:], s_rotated[1,:], s_rotated[2,:], plasticity)
    yields_tr = np.transpose(yields) # a 3xnl matrix
    p_a = plasticity*(s_rotated[0,:]-yields_tr[0,:])
    p_l = plasticity*(s_rotated[1,:]-yields_tr[1,:])
    p_sh = plasticity*(s_rotated[2,:]-yields_tr[2,:])

    # Compute the changes in wall stresses in the MF frame
    dsdt = p.C_el @ eps_MF_tr - 1/p.tau_visc*np.vstack((p_a, p_l, p_sh)) # gives a 3xnl stress changes matrix

    # Put it back in the reference frame
    Contra_Rotate = rotation_matrix_stress(MFA-np.pi/2*np.ones_like(MFA)) # a nlx3x3 rotation matrix

    dsdt_Ref = np.zeros((3, len(MFA)))    
    for i in range(len(MFA)):
      # Perform multiplication for each layer -> IT IS NOT VECTORIZED !!! =(
      dsdt_Ref[:, i] = Contra_Rotate[i, :, :] @ dsdt[:, i]
          
    # then extract stress changes for the relevant layers
    dsadt[0:iteration_count+1,0] = dsdt_Ref[0,:(iteration_count+1)] # anticlinal stress
    dsldt[0:iteration_count+1,0] = dsdt_Ref[1,:(iteration_count+1)] # longitudinal stress
    dtaudt[0:iteration_count+1,0] = dsdt_Ref[2,:(iteration_count+1)] # shear stress
        
    # Gather the derivative vector dy/dt
    dy = np.append([dnsdt,dLdt],dsadt)
    dy = np.append(dy,dsldt)
    dy = np.append(dy,dtaudt)
    dy = np.append(dy,dWadt)
    
    # return the vector of time changes
    return dy 