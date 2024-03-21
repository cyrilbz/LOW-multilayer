# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:59:24 2024

@author: cbozonnet
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import pickle
from functions import data2save, parameters

# List filenames and legends
list_files = ['test.pkl']
my_legend = ['test']
list_files = ['multi-high_viscosity.pkl']
list_files = ['multi-nl1.pkl','multi-nl20.pkl','multi-nl100.pkl','multi-nl400.pkl','multi-nl800.pkl']
my_legend = ['nl=1','nl=20','nl=100','nl=400','nl=800']
list_files = ['multi-nl800.pkl']
my_legend = ['nl=40']
# list_files = ['complete-Cs_ext200.pkl','complete-Cs_ext200-omegax2.pkl','complete-Cs_ext200-omegax0-osmoreg2.pkl']
# my_legend = ['Cs_ext=200','Cs_ext=200 omegax2','Cs_ext=200 omegax0']
my_color =['b','orange','green','red','magenta']
# list_files = ['complete-Cs_ext200.pkl']
# my_legend = ['Cs_ext=200']
# list_files = ['complete-Cs_ext200-long.pkl','complete-Cs_ext200-n_constant.pkl','complete-Cs_ext200-PI_constant.pkl']
# my_legend = ['Cs_ext=200','n=cste','PI=cste']
# list_files = ['complete-Cs_ext200-dt0_5h.pkl','complete-Cs_ext200-dt2h.pkl','complete-Cs_ext200-dt4h.pkl','complete-Cs_ext200-dt10h.pkl','complete-Cs_ext200-dt50h.pkl']
# my_legend = ['dt=0.5h','dt=2h','dt=4h','dt=10h','dt=50h']
# list_files = ['complete-Cs_ext200-dt4h.pkl']
# my_legend = ['dt=4h']
size = len(list_files) # get the number of files

for i in range(size): # loop to open the files one by one and plot things
    path = "../runs/" + list_files[i]
    with open(path, "rb") as file:
        data = pickle.load(file)
    sol = data.sol
    p = data.p
    t = data.t
    nt = len(t) # number of iterations
    
    ####### Post process #########
    # extract data from solution
    ns = sol[:,0]
    La = sol[:,1]
    sa = sol[:,2:2+p.nl]
    Wa = sol[:,2+p.nl:2+2*p.nl]
    
    # allocate some arrays for post processing
    sa_mean = np.zeros((nt,1)) 
    sp_mean = np.zeros((nt,1)) 
    P = np.zeros((nt,1)) 
    Q = np.zeros((nt,1)) 
    W_sum= np.zeros((nt,p.nl)) 
    
    # Data treatment
    th=t/3600 # time in hours
    WaT = np.sum(Wa, axis=1)
    WpT = p.Wp0
    VwaT = 2*WaT*p.Lz*(La-2*WpT)
    VwpT = 2*WpT*p.Lp*p.Lz
    Vh = La*p.Lp*p.Lz - VwpT- VwaT
    Cs= ns/Vh
    PI=Cs*p.Rg*p.T
    
    # Compute mean stresses and pressure
    for k in range(nt):
        sa_mean[k] = np.dot(sa[k],Wa[k])/WaT[k] # mean anticlinal stress
        P[k] = 2*sa_mean[k]*WaT[k]/p.Lp + p.P_ext # pressure  
        sp_mean[k] = (P[k]-p.P_ext)*La[k]/(2*WpT)
        # Compute the flow rate   
        A = 2*p.Lz*(p.Lp+La[k]) # area for water fluxes
        Q[k] = A*p.kh*(p.Psi_src-P[k]+PI[k]) # water fluxes
        W_sum[k]=np.cumsum(Wa[k])
    
        
    # # Compute Lockhart's solution
    # # NB: to match with it 
    # # you should take W -> 0 in the parameters
    # # In addition to osmoregulation & no wall synthesis
    # Py0 = p.sig_Y*2*p.W0/p.R0 # yield pressure
    # # corrected wall synthesis rate
    # phi_w_star = p.phi_w*p.R0/(2*p.W0)*(1+2*p.W0/p.R0)
    # # Compute growth rate
    # Gth_cor = phi_w_star*p.phi_h/(phi_w_star+p.phi_h)*(p.Psi_src+p.Pi0-Py0)
    # # exponential growth
    # L_th = p.L0*np.exp(Gth_cor*th*3600)
    
    ######### Plots ##########
    plt.figure(1)
    plt.plot(th,(La)*1000000,label=my_legend[i],color=my_color[i])
    # plt.plot(th,L_th**1000000,'--k')
    # plt.plot([150, 150], [0, 84],':b',linewidth=1.5)
    # plt.plot([90, 90], [0, 42],color='orange',linewidth=1.5,linestyle=':')
    # plt.xlim((0,300))
    # plt.ylim((0,150))
    plt.legend(loc='best')
    #plt.yscale('log')
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{R [$\mu$m]}", fontsize=16)
    plt.title(r"$\textbf{Cell radius}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(2)
    plt.plot(th,sa_mean/1e6,label=my_legend[i],color=my_color[i])
    plt.plot([0, p.t_end/3600], [1, 1],':k',linewidth=1.5)
    # plt.ylim((0,1.1))
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{$\overline{\sigma_a}$ [MPa]}", fontsize=16)
    plt.title(r"$\textbf{Mean anticlinal wall stress}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(3)
    plt.plot(th,P/1e6,color=my_color[i])
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{$P$ [MPa]}", fontsize=16)
    plt.title(r"$\textbf{Turgor pressure}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(4)
    plt.plot(th,WaT*1000000,label=my_legend[i],color=my_color[i])
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{Wa [$\mu$m]}", fontsize=16)
    plt.title(r"$\textbf{Wall thickness}$", fontsize=16)
    # Set grid and minor ticks
    plt.xlim((80,100))
    plt.ylim((3.5,4.5))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(5)
    plt.plot(th,PI/1e6,color=my_color[i])
    # plt.legend(loc='best')
    plt.xlabel('t [h]')
    plt.ylabel('PI [MPa]')
    
    plt.figure(6)
    plt.plot(th,sp_mean/1e6,label=my_legend[i],color=my_color[i])
    plt.plot([0, p.t_end/3600], [1, 1],':k',linewidth=1.5)
    # plt.ylim((0,1.1))
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{$\overline{\sigma_p}$ [MPa]}", fontsize=16)
    plt.title(r"$\textbf{Mean periclinal wall stress}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(7)
    plt.plot(th,sa/1e6)
    plt.plot([0, p.t_end/3600], [1, 1],':k',linewidth=1.5)
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{$\sigma_a$ [MPa]}", fontsize=16)
    plt.title(r"$\textbf{Stress changes for all layers}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(8)
    plt.plot(th,Q,label=my_legend[i],color=my_color[i])
    # plt.ylim((0,1.1))
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{$Q$ [m3/s]}", fontsize=16)
    plt.title(r"$\textbf{Flow rate}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(9)
    # Spatio-temporal stress profile
    # Define your desired colormap (e.g., 'viridis')
    cmap = plt.cm.viridis
    
    # Select equally distributed timesteps (adjust divisor for different numbers)
    selected_timesteps = np.linspace(10, nt-1, 8, dtype=int)  # Use np.linspace for even distribution
    
    # Create a color mapper object for the selected timesteps
    norm = plt.Normalize(vmin=selected_timesteps[0], vmax=selected_timesteps[-1])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    
    # Plot loop for each timestep
    for k,timestep in enumerate(selected_timesteps):
        # Get color based on timestep
        color = sm.to_rgba(timestep)
        
        # Get data points where wa is not zero (boolean mask)
        mask = Wa[timestep] != 0
        
        # Filter W_sum and sa based on the mask
        W_sum_f = W_sum[timestep][mask]
        sa_f = sa[timestep][mask]
        
        # plot at mean layer thickness values
        if len(W_sum_f) > 1:  # Check for at least two elements
            W_sum_f[0] = (W_sum_f[0]) / 2  # Average first element
            W_sum_f[-1] = (W_sum_f[-1] + W_sum_f[-2]) / 2  # Average last element

        # Local average for non-edge elements (vectorized)
        W_sum_f[1:-1] = (W_sum_f[:-2] + W_sum_f[2:]) / 2


        # Plot with color from colormap
        plt.plot(W_sum_f*1e6, sa_f*1e-6, color=color, label=f'Time: {th[timestep]}',marker='.')
        
    #plt.plot(W_sum[-1]*1e6,sa[-1]/1e6,color=my_color[i])
    # plt.ylim((0,1.1))
    plt.xlabel(r"\textbf{Position within the wall [µm]}", fontsize=16)
    plt.ylabel(r"\textbf{$\sigma_a$ [MPa]}", fontsize=16)
    plt.title(r"$\textbf{Anticlinal stress profiles}$", fontsize=16)
    plt.legend(loc='best')
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()