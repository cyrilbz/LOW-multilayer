# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:40:39 2024

@author: cbozonnet
"""

# This is a program for a Lockhart/Ortega model
# WITH A MULTILAYER WALL & MICROFIBRILS
# with elongation
# with wall thickening related to sugar availability
# with sugar uptake
# all parameters are in function.py

import numpy as np
from scipy.integrate import odeint, solve_ivp
import pickle
from functions import data2save, parameters, rotation_matrix_strain, rotation_matrix_stress, compute_thresholds
import time
import os
import sys
import itertools

############### Specify simulations parameters and output names

# simulation parameters
overwrite = True # True if you want to overwrite the existing same file 
param_study = False# True if you want to run a parametric study 
folder_path = "../runs/" # folder in which you want to write the results
number_parameter = 2 # for param_study=True ; 1 or 2 - more is not implemented so far
name_1 = 'MFA0_deg' # parameter name as listed in functions.py
value_1 = np.arange(0,95,5) # parameter values
name_2 = 'eps0' # parameter name as listed exactly in functions.py
value_2 = [0.02] # parameter values

# handle simulation number and output files
file_list = []
if param_study==True: ##### in case of parametric study ######
    match number_parameter:
        case 1: # only one parameter
            pattern = r"effect_MFA_fine" # pattern for the file name
            for value in value_1:
                filename = f"{pattern}{int(value)}.pkl" 
                file_list.append(filename)
        case 2: # two parameters
            # List all possible combinations 
            combinations = list(itertools.product(value_1 , value_2))    
            print ("There are " +str(len(combinations))+" simulations to run.")
            for i in range(len(combinations)):
                filename = name_1+'-'+f"{combinations[i][0]:.3f}"+'-'+name_2+'-'+f"{combinations[i][1]:.3f}"+'.pkl'
                file_list.append(filename)
        case _: # all other values
            print("Param number not handled currently")
else:           
    # Specify the output filename for a single simulation
    filename ='multi-classic.pkl'
    file_list.append(filename)
    

######### nothing to change below
    
def dydt(t,y,p): # all physical equations for time integration
    # NB : to switch from odeint to solve_ivp and vice versa
    # you must switch t and y in the previous line
    # and comment/uncomment the call to the solver below
    
    # get global variables - useful to handle the multilayer approach
    global iteration_count, tdepo, Ldepo, t_pressure, k_pressure, theta_depo
    
    # extract data from y vector
    ns = y[0] # extract sugar content
    La = y[1] # extract cell length
    sa = y[2:2+p.nl] # extract anticlinal stress
    sl = y[2+p.nl:2+2*p.nl] # extract longitudinal stress
    tau = y[2+2*p.nl:2+3*p.nl] # extract longitudinal stress
    Wa = y[2+3*p.nl:2+4*p.nl] # extract anticlinal wall thickness
    theta_MT = y[-1] # extract microtubules angle
    
    # Initialize time derivatives
    dnsdt = 0
    dLdt = 0
    dsadt = np.zeros((p.nl,1))
    dsldt = np.zeros((p.nl,1)) 
    dtaudt = np.zeros((p.nl,1)) 
    dWadt = np.zeros((p.nl,1))
    dtheta_MTdt = 0
    
    # Initialize some other vectors
    eps_t = np.zeros((p.nl)) # total deformation
    MFA = np.zeros((p.nl)) # micro fibril angles
    
    # Go to next layer
    # if time is sufficiently high and we did not yet reach the max number of layers (>1)
    if (tdepo[iteration_count]<t and iteration_count < len(tdepo)-1 and p.nl>1):
        iteration_count += 1 # switch to next layer
        Ldepo[iteration_count] = La # save deposition length
        theta_depo[iteration_count] = theta_MT # save MT angle at deposition time
        
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
    sl_mean = np.dot(sl,Wa)/WaT # mean longitudinal stress
    tau_mean = np.dot(tau,Wa)/WaT # mean longitudinal stress
    P = 2*sa_mean*WaT/(p.Lp-2*WaT) + p.P_ext # pressure
    
    # Compute stress eigenvector direction
    Rz = np.sqrt(((sa_mean-sl_mean)/2)**2+tau_mean**2) # Mohr circle center position
    theta_p = np.pi/2-np.arctan(tau_mean/(Rz+(sa_mean-sl_mean)/2)) # Eigenvector direction
    
    # compute change in microtubules angle
    if (p.change_deposition==True):
        #dtheta_MTdt = (theta_p - theta_MT)/p.tau_MT # adjustement towards stress eigenvector
        dtheta_MTdt = (p.theta_MT_max -p.MFA0)/p.t_end # progressive linear shift
        
    # Compute the total deformation and MFA related quantitites
    eps_t[:(iteration_count+1)] = (La - Ldepo[:(iteration_count+1)])/(Ldepo[:(iteration_count+1)]-2*WpT)    
    if (p.no_rotation==True): 
        MFA = np.arctan(np.tan(theta_depo)*np.exp(np.zeros((p.nl)))) # force no rotation
    else: 
        MFA = np.arctan(np.tan(theta_depo)*np.exp(eps_t)) # MFA evolution

    # Compute water fluxes
    my_psiX = p.Psi_src + 0.5*p.delta_PsiX*(1+np.cos((t/3600+12-0)*np.pi/12))
    A = 2*p.Lz*(p.Lp+La) # area for water fluxes
    Q = A*p.kh*(my_psiX-P+PI+p_forcing) # water fluxes
    
    # compute the anticlinal elongation rate (ERa)
    ERa = (1-2*WaT/p.Lp)*(Q+dVwadt)/Vh
    if (p.wall_regul==True): ERa = Q/Vh # wall thickness regulation
    if (p.force_elongation == True): ERa = p.G_forced

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
    # p_a = plasticity*np.maximum(s_rotated[0,:]-yields_tr[0,:],np.zeros((p.nl)))
    # p_l = plasticity*np.maximum(s_rotated[1,:]-yields_tr[1,:],np.zeros((p.nl)))
    # p_sh = plasticity*np.maximum(s_rotated[2,:]-yields_tr[2,:],np.zeros((p.nl)))
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
    dy = np.append(dy,dtheta_MTdt)
    
    # return the vector of time changes
    return dy 

######### Main program ##########

for sim in range(len(file_list)): # loop over all simulations
    start_time = time.time()    
    path = folder_path + file_list[sim] 
    print("Simulation running: " + str(file_list[sim]))
    if os.path.exists(path) and overwrite==False: # Check if the file already exists
        print("This output file already exists.") 
        continue

    # get all parameters
    p = parameters() # get all parameters 
    
    # replace the new parameter(s) in case of parametric study
    if param_study==True:
        match number_parameter:
            case 1:
                p.set_param(name_1,value_1[sim])
            case 2:
                p.set_param(name_1,combinations[sim][0])
                p.set_param(name_2,combinations[sim][1])

    # Intial values
    t = np.arange(p.t0,p.t_end+p.dt,p.dt) # time vector
    sa0 = np.ones(p.nl) # all layers with the same initial stress
    sl0 = np.zeros(p.nl) # initial longitudinal stress
    tau0 = np.zeros(p.nl) # initial tangential stress
    Wa0 = np.zeros(p.nl) # intialize all layers with zero thickness
    Wa0[0] = p.Wa0 # initial first layer
    # initial microtubules angle regarding the horizontal direction
    theta_MT = p.MFA0_deg*np.pi/180 
    
    
    y0 = [p.ns0,p.La] # initial values 
    y0 = np.append(y0, sa0) # combine
    y0 = np.append(y0, sl0) # combine
    y0 = np.append(y0, tau0) # combine
    y0 = np.append(y0,Wa0) # combine
    y0 = np.append(y0,theta_MT)
    
    # Handle multilayer aproach
    iteration_count = 0 # Initialize global variable
    tdepo = np.linspace(p.t0+p.tfirstlayer, p.t_end, p.nl, endpoint=True) # target time values for new layer creation
    Ldepo = np.zeros(p.nl) # vector to store all deposition length to compute total deformation
    theta_depo = np.ones(p.nl)*(p.MFA0_deg*np.pi/180) # initial deposition angle
    Ldepo[0] = p.La # first layer already deposited
    yields_tr = np.zeros((3,p.nl))
    
    # Handle pressure forcing
    k_pressure=0
    t_pressure=0
    if (p.pressure_steps==True):
        t_pressure = np.arange(p.t0,p.t_end+p.dt_plateau,p.dt_plateau)
    
    # Resolution    
    sol = solve_ivp(dydt, [p.t0 , p.t_end+p.dt], y0, args=(p,),t_eval=t, \
                    method='RK45',max_step=p.dt) 
    # sol = odeint(dydt, y0, t, args=(p,)) 
      #, atol = 1e-10 , rtol = 1e-6
    ######## Store solution #########
    data = data2save(p,t,sol,tdepo,Ldepo,theta_depo) # create data structure
    with open(path, "wb") as file: # open file
        pickle.dump(data, file) # save data
        
    #print(theta_depo*180/np.pi)
    
    # Calculate the elapsed time.
    elapsed_time = time.time() - start_time
    
    # Display the elapsed time to the user.
    print("The elapsed time for this run is {} seconds.".format(elapsed_time))