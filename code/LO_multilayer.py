# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:40:39 2024

@author: cbozonnet
"""

# This is a program for a Lockhart/Ortega model
# WITH A MULTILAYER WALL
# with elongation
# with wall thickening related to sugar availability
# with sugar uptake
# all parameters are in function.py

import numpy as np
from scipy.integrate import odeint, solve_ivp
import pickle
from functions import data2save, parameters
import time
import os

# Start the timer
start_time = time.time()

# Specify the output filename
filename ='W-cste-nl40-E10-high_conductivity.pkl'
overwrite = True # True if you want to overwrite the existing file 
path = "../runs/" + filename

if os.path.exists(path) and overwrite==False: # Check if the file already exists
    raise Exception("This output file already exists.") 
   
print("Simulation running: " + str(filename))

def dydt(y,t,p): # all physical equations for time integration
    # NB : to switch from odeint to solve_ivp 
    # you must switch t and y in the previous line
    
    # get global variables - useful to handle the multilayer approach
    global iteration_count, t_target  
    
    # handle the switch from one layer to another
    if (t_target[iteration_count]<t and iteration_count < len(t_target)-1 and p.nl>1):
        iteration_count += 1 # switch to next layer

    # extract data from y vector
    ns = y[0] # extract sugar content
    La = y[1] # extract cell length
    sa = y[2:2+p.nl] # extract anticlinal stress
    Wa = y[2+p.nl:2+2*p.nl+1] # extract anticlinal wall thickness

    # Initialize time derivatives
    dnsdt = 0
    dLdt = 0
    dsadt = np.zeros((p.nl,1)) 
    dWadt = np.zeros((p.nl,1))
    
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
    
    # Compute water fluxes
    my_psiX = p.Psi_src + 0.5*p.delta_PsiX*(1+np.cos((t/3600+12-0)*np.pi/12))
    A = 2*p.Lz*(p.Lp+La) # area for water fluxes
    Q = A*p.kh*(my_psiX-P+PI) # water fluxes
    
    # compute the anticlinal elongation rate (ERa)
    ERa = (Q+dVwadt)/(Vh)
    #ERa = Q/Vh*1/(1-2*WaT/p.Lp) # wall thickness regulation
    
    # Compute the anticlinal length changes
    dLdt=(La-2*WpT)*ERa
    
    # Wall synthesis when thickness regulation
    #dVwadt=2*p.Lz*WaT*dLdt
    
    # Compute the changes in wall layer thicknesses
    dWadt = -Wa*1/(La-2*WpT)*(dLdt) # thickness evolution for all layers
    dWadt[iteration_count] += dVwadt/(2*p.Lz*(La-2*WpT)) # wall synthesis in a target layer
    #dWadt[0] += dVwadt/(2*p.Lz*(La-2*WpT)) # wall synthesis in a target layer
    
    # compute changesin sugar content 
    #dnsdt = -1/p.MMs*dMdt + fs # no osmoregulation
    dnsdt = ns/Vh*(dLdt*(p.Lp*p.Lz-2*WaT*p.Lz)-2*p.Lz*sum(dWadt)*(La-2*WpT)) # omoregulation
                
    # Compute the changes in wall stresses
    sa_subset = sa[:(iteration_count+1)]
    plasticity = np.maximum(sa_subset-p.sig_Y,np.zeros((1,iteration_count+1)))
    plasticity = plasticity.reshape((iteration_count+1, 1))  # Reshape to have one column
    dsadt[0:iteration_count+1] = p.E*ERa -p.E/p.mu*plasticity
    #dsadt[0] = p.E*ERa-p.E/p.mu*max(sa[0]-p.sig_Y,0)
    
    # Gather the derivative vector dy/dt
    dy = np.append([dnsdt,dLdt],dsadt)
    dy = np.append(dy,dWadt)
    
    # return the vector of time changes
    return dy 

######### Main program ##########
p = parameters() # get all parameters       
t = np.arange(p.t0,p.t_end+p.dt,p.dt) # time vector
t_target = np.linspace(p.t0, p.t_end, p.nl, endpoint=True) # target time values for new layer creation
sa0 = p.sig_a0*np.ones(p.nl) # all layers with the same initial stress
Wa0 = np.zeros(p.nl) # intialize all layers with zero thickness
Wa0[0] = p.Wa0 # initial first layer
y0 = [p.ns0,p.La] # initial values 
y0 = np.append(y0,sa0) # combine
y0 = np.append(y0,Wa0) # combine
iteration_count = 0 # Initialize global variable
# sol = solve_ivp(dydt, [p.t0 , p.t_end+p.dt], y0, tspan=t,args=(p,)) # resolution
sol = odeint(dydt, y0, t, args=(p,)) # resolution

######## Store solution #########
data = data2save(p,t,sol) # create data structure
with open(path, "wb") as file: # open file
    pickle.dump(data, file) # save data

# Calculate the elapsed time.
elapsed_time = time.time() - start_time

# Display the elapsed time to the user.
print("The elapsed time is {} seconds.".format(elapsed_time))