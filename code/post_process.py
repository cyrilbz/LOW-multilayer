# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:59:24 2024

@author: cbozonnet
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.integrate import odeint, solve_ivp
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
list_files = ['multi-nl1-corrected_P.pkl','multi-nl40-corrected_P.pkl']
my_legend = ['nl=1','nl=40']
list_files = ['multi-nl200-corrected_P-normal_synth.pkl','multi-nl200-corrected_P-normal_synth-low_E.pkl']
my_legend = ['E=100MPa','E=10MPa']
list_files = ['multi-nl40-E10MPa-100h-Lp50um-test_osmoreg-test_wallreg-higher_ext.pkl']
my_legend = ['E=10MPa']
list_files = ['W-cste-nl1.pkl','W-cste-nl40.pkl','W-cste-nl100.pkl','W-cste-nl200.pkl']
my_legend = ['1l','40l','100l','200l']
# list_files = ['nl1.pkl','nl40.pkl','nl100.pkl','nl200.pkl']
# my_legend = ['1l','40l','100l','200l']
# list_files = ['nl1-E10.pkl','nl40-E10.pkl']
# my_legend = ['1l','40l','100l','200l']
# list_files = ['nl1-test_cycle-0.5MPa.pkl','nl40-test_cycle-0.5MPa.pkl']
# my_legend = ['nl=1','nl=40']
# list_files = ['complete-Cs_ext200.pkl','complete-Cs_ext200-omegax2.pkl','complete-Cs_ext200-omegax0-osmoreg2.pkl']
# my_legend = ['Cs_ext=200','Cs_ext=200 omegax2','Cs_ext=200 omegax0']
my_color =['b','orange','magenta','red','green']
# list_files = ['multi-nl320-E10MPa-800h.pkl']
# my_legend = ['Cs_ext=200']
list_files = ['Lockhart-mux10.pkl','Lockhart-mux10-nl40.pkl']
list_files = ['W_variable-mux1-nl40.pkl','W_variable-mux100-nl40.pkl','W_variable-mux1000-nl40.pkl','W_variable-mux10000-nl40.pkl']
list_files = ['W_variable-khx1.pkl','W_variable-khx5.pkl','W_variable-khx10.pkl','W_variable-khx100.pkl']
# my_legend = ['1 l','40l','100l','200l']
#list_files = ['Lockhart-khx1.pkl','Lockhart-khx10.pkl','Lockhart-khx100.pkl']
#my_legend = ['Khx1','Khx2','Khx10']
list_files = ['test_tancrede.pkl']

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
    #sol = data.sol.y # get solution
    sol = data.sol # get solution
    #sol = np.transpose(sol)
    p = data.p # get parameters
    t = data.t # get time vector
    tdepo = data.tdepo # get deposition time vector
    Ldepo= data.Ldepo
    nt = len(t) # number of iterations
    my_legend[i] = f"{p.alpha:.2f}"
    
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
    G = np.zeros((nt,1)) 
    my_psiX = np.zeros((nt,1)) 
    W_sum = np.zeros((nt,p.nl)) 
    eps_t = np.zeros((p.nl,nt)) # prepare space for total deformation 
    s_MF = np.zeros((nt,p.nl))
    tau_MF = np.zeros((nt,p.nl))
    
    # Data treatment
    th=t/3600 # time in hours
    WaT = np.sum(Wa, axis=1)
    WpT = p.Wp0
    VwaT = 2*WaT*p.Lz*(La-2*WpT)
    VwpT = 2*WpT*p.Lp*p.Lz
    Vh = La*p.Lp*p.Lz - VwpT- VwaT
    Cs = ns/Vh
    PI = Cs*p.Rg*p.T
        
    # Compute areas
    Cell_area = La*p.Lp
    Lumen_area = Vh/p.Lz
    Wall_area = (VwpT + VwaT)/p.Lz
    check = Wall_area+Lumen_area
    
    # Compute mean stresses and pressure
    for k in range(nt):
        sa_mean[k] = np.dot(sa[k],Wa[k])/WaT[k] # mean anticlinal stress
        P[k] = 2*sa_mean[k]*WaT[k]/(p.Lp-2*WaT[k]) + p.P_ext # pressure  
        sp_mean[k] = (P[k]-p.P_ext)*(La[k]-2*WpT)/(2*WpT)
        # Compute the flow rate   
        A = 2*p.Lz*(p.Lp+La[k]) # area for water fluxes
        # my_psiX[k] = p.Psi_src + 0.5*(p.delta_PsiX)*(1+np.cos((t[k]/3600+12-0)*np.pi/12))
        my_psiX[k] = p.Psi_src
        Q[k] = A*p.kh*(my_psiX[k]-P[k]+PI[k]) # water fluxes
        W_sum[k]=np.cumsum(Wa[k]) # total wall thickness
        G[k] = Q[k]/Vh[k] # wall synthesis regulation
        dMmax = p.omega*Vh[k]*np.exp(p.Eaw/p.kb*(1/p.T0-1/p.T))
        dVaTdt =  dMmax/p.rho_w*Cs[k]/(Cs[k]+p.Km)
        if (p.wall_regul==False):
            G[k] = (1-2*WaT[k]/p.Lp)*(Q[k]+dVaTdt)/Vh[k] # G with synthesis non regulated
        
        
    for k in range(p.nl): # move across all layers
        if (k==0):
            mask = (t>=0) # filter when the layer has not been created
        else:
            mask = (t>tdepo[k-1]) # filter when the layer has not been created
        eps_t[k][mask] = (La[mask]-Ldepo[k])/Ldepo[k]  # compute total deformation

    eps_t = np.transpose(eps_t) # transpose the matrix to be consistent with other matrix      
    phi0 = 5*np.pi/180 # MFA at deposition
    MFA = np.arctan(np.tan(phi0)*np.exp(eps_t)) # micro-fibril angle evolution
    
    for k in range(p.nl):
        s_MF[:,k] = sa[:,k]/2*(1+np.cos(2*MFA[:,k])) # traction in the MF axis
        tau_MF[:,k] = -sa[:,k]/2*np.sin(2*MFA[:,k]) # shear in the MF frame

    # # Compute the theoretical (Lockhart like) solution
    # def lockhart_cambium(y,t,p):
    #     Lth = y # cell length
    #     PM = p.Psi_src + p.Pi0 # motor power
    #     Py0 = p.sig_Y*2*p.Wa0/(p.Lp-2*p.Wa0) # yield pressure
    #     phi_w = 1/p.mu*(p.Lp-2*p.Wa0)/(2*p.Wa0)
    #     phi_a = 2*(Lth + p.Lp)/((Lth-2*p.Wp0)*(p.Lp-2*p.Wa0))*p.kh
    #     alpha = phi_a/(phi_a+phi_w)
    #     dydt = (Lth-2*p.Wp0)*alpha*phi_w*(PM-Py0)
    #     return dydt 
    
    # lock = odeint(lockhart_cambium, p.La, t, args=(p,))
    
    # Compute pressure and growth rate using (adapted) Lockhart solution
    PM = p.Psi_src + p.Pi0 # motor power
    # Py0 = p.sig_Y*2*p.Wa0/(p.Lp-2*p.Wa0) # yield pressure
    # phi_w = 1/p.mu*(p.Lp-2*p.Wa0)/(2*p.Wa0)
    # phi_a = 2*(La + p.Lp)/((La-2*p.Wp0)*(p.Lp-2*p.Wa0))*p.kh
    Py0 = p.sig_Y*2*WaT/(p.Lp-2*WaT) # yield pressure
    phi_w = 1/p.mu*(p.Lp-2*WaT)/(2*WaT) # no synthesis case
    #phi_w = 1/p.mu*(p.Lp)/(2*WaT) # synthesis case
    phi_a = 2*(La + p.Lp)/((La-2*p.Wp0)*(p.Lp-2*WaT))*p.kh
    alpha = phi_a/(phi_a+phi_w)
    P_lock = alpha*PM + (1-alpha)*Py0 # no synthesis case
    #P_lock = alpha*PM + (1-alpha)*Py0 + p.omega/p.rho_w/(phi_a+phi_w)*1/(1+p.Rg*p.T*p.Km/PI)# synthesis case
    G_lock = alpha*phi_w*(PM-Py0) #+ p.omega/p.rho_w*alpha*1/(1+p.Rg*p.T*p.Km/PI)
    
    ######### Plots ##########
    plt.figure(1)
    plt.plot(th,(La)*1000000,label=my_legend[i],color=my_color[i])
    # plt.plot([150, 150], [0, 84],':b',linewidth=1.5)
    # plt.plot([90, 90], [0, 42],color='orange',linewidth=1.5,linestyle=':')
    # plt.xlim((0,100))
    # plt.ylim((0,1000))
    plt.legend(loc='best',title=r"$\alpha_0$")
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
    plt.legend(loc='best',title=r"$\alpha_0$")
    plt.plot([0, p.t_end/3600], [p.sig_Y/1e6, p.sig_Y/1e6],':k',linewidth=1.5)
    # plt.ylim((1,1.5))
    # plt.xlim((20,60))
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
    plt.plot(th,P/1e6,color=my_color[i],label=my_legend[i])    
    # plt.plot(th,P_lock/1e6,marker='v', markevery=8, linestyle='',color=my_color[i])
    plt.legend(loc='best',title=r"$\alpha_0$")
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
    plt.legend(loc='best')
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{Wa [$\mu$m]}", fontsize=16)
    plt.title(r"$\textbf{Wall thickness}$", fontsize=16)
    # Set grid and minor ticks
    # plt.xlim((80,100))
    # plt.ylim((3.5,4.5))
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
    
    #plt.figure(9)
    fig, ax = plt.subplots()
    # Spatio-temporal stress profile
    # Define your desired colormap (e.g., 'viridis')
    cmap = plt.cm.viridis
    
    # Select equally distributed timesteps (adjust divisor for different numbers)
    selected_timesteps = np.linspace(10, nt-1, 5, dtype=int)  # Use np.linspace for even distribution
    
    # Create a color mapper object for the selected timesteps
    norm = plt.Normalize(vmin=th[selected_timesteps[0]], vmax=th[selected_timesteps[-1]])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    
    # Plot loop for each timestep
    for k,timestep in enumerate(selected_timesteps):
        # Get color based on timestep
        color = sm.to_rgba(th[timestep])
        
        # Get data points where wa is not zero (boolean mask)
        mask = Wa[timestep] != 0
        
        # Filter W_sum and sa based on the mask
        W_sum_f = W_sum[timestep][mask]
        sa_f = sa[timestep][mask]
                
        # # Local average for non-edge elements (vectorized)
        # W_sum_f[1:-1] = (W_sum_f[:-2] + W_sum_f[2:]) / 2
        
        # # plot at mean layer thickness values
        # if len(W_sum_f) > 1:  # Check for at least two elements
        #     W_sum_f[0] = (W_sum_f[0]) / 2  # Average first element
        #     W_sum_f[-1] = (W_sum_f[-1] + W_sum_f[-2]) / 2  # Average last element
        # else:
        #     W_sum_f[0] = (W_sum_f[0]) / 2 

        # Plot with color from colormap
        ax.plot(W_sum_f*1e6, sa_f*1e-6, color=color, label=f'Time: {th[timestep]}',marker='.')
        
    # Add colorbar
    fig.colorbar(sm, label='Time [h]', ax=ax)  # Add colorbar to the axes    
    #plt.plot(W_sum[-1]*1e6,sa[-1]/1e6,color=my_color[i])
    # plt.ylim((0,1.1))
    plt.plot([p.Wa0/(1+p.sig_Y/p.E)*1e6, p.Wa0/(1+p.sig_Y/p.E)*1e6], [0, np.max(np.max(sa, axis=1))/1e6],':k',linewidth=1.5)
    plt.xlabel(r"\textbf{Position within the wall [µm]}", fontsize=16)
    plt.ylabel(r"\textbf{$\sigma_a$ [MPa]}", fontsize=16)
    plt.title(r"$\textbf{Anticlinal stress profiles}$", fontsize=16)
    #plt.legend(loc='best')
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    dpi = 300  # Dots per inch
    # Save the figure as a high-resolution JPEG
    #plt.savefig("stress_profiles-W_cste.jpeg", dpi=dpi)
    
    plt.figure(10)
    plt.plot(th,Cell_area*1e12,label='Cell area')
    plt.plot(th,Lumen_area*1e12,label='Lumen area')
    plt.plot(th,Wall_area*1e12,label='Wall area')
    # plt.ylim((0,1.1))
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{Areas in $\mu$m$^2$}", fontsize=16)
    plt.title(r"$\textbf{Areas}$", fontsize=16)
    plt.legend(loc='best')
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(11)
    plt.plot(th,eps_t*100)
    # plt.ylim((0,1.1))
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{$\varepsilon_T$ $[\%]$}", fontsize=16)
    plt.title(r"$\textbf{Total deformation in \%}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(12)
    plt.plot(th,MFA*180/np.pi)
    # plt.ylim((0,1.1))
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{MFA in degrees}", fontsize=16)
    plt.title(r"$\textbf{MFA changes for $\phi_0$="+str(p.MFA0_deg)+" degrees}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(13)
    plt.hist(MFA[-1]*180/np.pi,bins=20,density=True)
    # plt.ylim((0,1.1))
    plt.xlabel(r"\textbf{MFA range}", fontsize=16)
    plt.ylabel(r"\textbf{PDF}", fontsize=16)
    plt.title(r"$\textbf{MFA final distribution}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(14)
    plt.plot(th,s_MF[:,0]/1e6,label='Tensile stress in the MF frame')
    plt.plot(th,tau_MF[:,0]/1e6,label='Shear stress in the MF frame',linestyle=':')
    plt.plot(th,sa[:,0]/1e6,label='Tensile stress in the reference frame',linestyle='--')
    # plt.ylim((0,1.1))
    plt.legend(loc='best')
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{Stress [MPa]}", fontsize=16)
    plt.title(r"$\textbf{Stresses in the different frames}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(15)
    plt.plot(th,G*3600,label=my_legend[i],color=my_color[i])
    plt.plot(th,G_lock*3600,marker='v', markevery=8, linestyle='',color=my_color[i])
    # plt.ylim((0,1.1))
    plt.yscale('log')
    plt.legend(loc='best',title=r"$\alpha_0$")
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{Growth rate [1/h]}", fontsize=16)
    plt.title(r"$\textbf{Growth rate}$", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()