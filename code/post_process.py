# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:59:24 2024

@author: cbozonnet
"""

import sys
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import odeint, solve_ivp
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import pickle
from functions import*


myfigsize = (15/2.54, 10/2.54)

################## change everything you need here #######################

parametric_study = False # to launch post_process on a parametric study
skip_plot = False # to skip the plots
multilayer_study = False # to compute physical properties in multilayer case

############# No parametric study ################

# you may list several filenames to compare between cases

list_files =['test-monolayer.pkl','test-multilayer.pkl','test-multilayer-80.pkl']
list_files = ['test-fully_coupled-10l.pkl']
list_files = ['MFA0_deg-75.000-Em-10000000.000.pkl']
list_files = ['multi-G_forced-10.000-Vf-0.100.pkl','multi-G_forced-8.000-Vf-0.100.pkl']
list_files = ['multi-classic.pkl']
list_files = ['multi-with_rotation_change_deposition.pkl']
#list_files = ['test-rotation_only-80l.pkl']
# list_files = ['10l-G8e-7.pkl','30l-G8e-7.pkl','50l-G8e-7.pkl','80l-G8e-7.pkl','120l-G8e-7.pkl','180l-G8e-7.pkl','260l-G8e-7.pkl']
# list_files = ['test80l.pkl','test280l.pkl']
# list_files = ['test-low_MFA-high_AR.pkl']
# my_legend = ['1 layer','40 layers','80 layers','120 layers']
my_legend = ['classic']
# my_legend = ['30deg ','60','90deg']
legend_title = r"$MFA_0 (\circ)$"
legend_title = r"$\dot{\varepsilon_a}$"

#############  Parametric_study ################

save_data_parametric = True # to save some parametric study results (!! CHANGE THE FILE NAME BELOW !!)
read_data_file = False # to open some additional data to get computed mechanical constants

parameter_name = ['MFA0_deg','eps0']  # list of parameters to save along with the results
#parameter_name = ['G_forced','Vf']  #
# filename pattern for parametric study
pattern = r"MFA0_deg-([-\d.]+)-eps0-([-\d.]+).pkl"  
folder = "../runs/"

data_file = "effect_eps0.pkl" # data file to write

data2open = ["data_effect_MFA-lowdt.pkl"] # data file to open

################## nothing to change below #######################

if parametric_study==True: 
    files = []
    for file in os.listdir(folder):
        match = re.match(pattern, file)
        if match:
            files.append(file) 
    list_files = files #sort_files(files, pattern)

my_color = ['blue','red','orange','green','magenta'] # list color for plots
size = len(list_files) # get the number of files
lines = ["-","--","-."]

print ("There are " +str(size)+" simulations to analyse.")

# initialize data to be saved
sY = np.zeros((size))
E = np.zeros((size))
epsY = np.zeros((size))
tau_visc_save = np.zeros((size))
param2save = np.zeros((size,len(parameter_name)))
if multilayer_study==True:
    sigma_saved = np.zeros((size))

if skip_plot==False: # define some elements for the plot with an inset
    fig= plt.figure(12,figsize=myfigsize)
    ax = fig.add_axes([0,0,1,1])
    # axins = fig.add_axes([0.635,0.3,0.3,0.3])  # Location of the inset

for i in range(size): # loop to open the files one by one and plot things
    print("Simulation running: " + str(list_files[i]))
    path = "../runs/" + list_files[i]
    with open(path, "rb") as file:
        data = pickle.load(file)
    sol = data.sol.y # get solution
    # sol = data.sol # get solution
    sol = np.transpose(sol)
    p = data.p # get parameters
    t = data.t # get time vector
    tdepo = data.tdepo # get deposition time vector
    Ldepo = data.Ldepo
    if hasattr(data, 'theta_depo'):
        theta_depo = data.theta_depo
    else:
        theta_depo = p.MFA0_deg*np.pi/180 # MFA at deposition
    nt = len(t) # number of iterations
    #my_legend[i] = f"{p.MFA0_deg:.2f}"

    ####### Post process #########
    # extract data from solution
    ns = sol[:,0]
    La = sol[:,1]
    sa = sol[:,2:2+p.nl]
    sl = sol[:,2+p.nl:2+2*p.nl] # extract longitudinal stress
    tau = sol[:,2+2*p.nl:2+3*p.nl] # extract tangential stress
    Wa = sol[:,2+3*p.nl:2+4*p.nl]
    theta_MT = sol[:,-1]
    
    # allocate some arrays for post processing
    sa_mean = np.zeros((nt,1)) 
    sp_mean = np.zeros((nt,1)) 
    sl_mean = np.zeros((nt,1)) 
    tau_mean = np.zeros((nt,1)) 
    P = np.zeros((nt,1)) 
    Q = np.zeros((nt,1)) 
    G = np.zeros((nt,1)) 
    my_psiX = np.zeros((nt,1)) 
    W_sum = np.zeros((nt,p.nl)) 
    eps_t = np.zeros((p.nl,nt)) # prepare space for total deformation
    MFA = np.zeros((p.nl,nt))
    MFA_mean = np.zeros((nt,1)) 
    MFA_median = np.zeros((nt,1))
    yields_rotated = np.zeros((3,p.nl,nt))
    k_pressure=0
    p_forcing = np.zeros((len(t)))
    t_pressure=0
    if (p.pressure_steps==True):
        t_pressure = np.arange(p.t0,p.t_end+p.dt_plateau,p.dt_plateau)
    Tsai = np.zeros((p.nl,nt))
    
    # Data treatment
    th=t/3600 # time in hours
    WaT = np.sum(Wa, axis=1)
    WpT = p.Wp0
    VwaT = 2*WaT*p.Lz*(La-2*WpT)
    VwpT = 2*WpT*p.Lp*p.Lz
    Vh = La*p.Lp*p.Lz - VwpT- VwaT
    Cs = ns/Vh
    PI = Cs*p.Rg*p.T
    theta_MT = theta_MT*180/np.pi
        
    # Compute areas
    Cell_area = La*p.Lp
    Lumen_area = Vh/p.Lz
    Wall_area = (VwpT + VwaT)/p.Lz
    check = Wall_area+Lumen_area

    ###############################################
    ############## Recompute temporal changes
    for k in range(nt):
        sa_mean[k] = np.dot(sa[k],Wa[k])/WaT[k] # mean anticlinal stress
        sl_mean[k] = np.dot(sl[k],Wa[k])/WaT[k]
        tau_mean[k] = np.dot(tau[k],Wa[k])/WaT[k]
        P[k] = 2*sa_mean[k]*WaT[k]/(p.Lp-2*WaT[k]) + p.P_ext # pressure  
        sp_mean[k] = (P[k]-p.P_ext)*(La[k]-2*WpT)/(2*WpT)
        # Compute the flow rate   
        A = 2*p.Lz*(p.Lp+La[k]) # area for water fluxes
        # my_psiX[k] = p.Psi_src + 0.5*(p.delta_PsiX)*(1+np.cos((t[k]/3600+12-0)*np.pi/12))
        my_psiX[k] = p.Psi_src
        # handle pressure forcing
        if (p.pressure_steps==True):
            if (k_pressure < len(t_pressure)-1): 
                if (t_pressure[k_pressure+1]<t[k]): k_pressure +=1 # update increment        
            p_forcing[k] = p.p_init + k_pressure*p.deltaP # compute pressure forcing
        Q[k] = A*p.kh*(my_psiX[k]-P[k]+PI[k]+p_forcing[k]) # water fluxes
        W_sum[k]=np.cumsum(Wa[k]) # total wall thickness
        G[k] = Q[k]/Vh[k] # wall synthesis regulation
        dMmax = p.omega*Vh[k]*np.exp(p.Eaw/p.kb*(1/p.T0-1/p.T))
        dVaTdt =  dMmax/p.rho_w*Cs[k]/(Cs[k]+p.Km)
        if (p.wall_regul==False):
            G[k] = (1-2*WaT[k]/p.Lp)*(Q[k]+dVaTdt)/Vh[k] # G with synthesis non regulated
        if (p.force_elongation == True): 
            G[k] = p.G_forced

    for k in range(p.nl):
        if (k==0):
            mask = (t>=0) # filter when the layer has not been created
        else:
            mask = (t>tdepo[k-1]) # filter when the layer has not been created
        eps_t[k][mask] = (La[mask]-Ldepo[k])/(Ldepo[k]-2*WpT)  # compute total deformation
            
    eps_t = np.transpose(eps_t) # transpose the matrix to be consistent with other matrix      
    phi0 = p.MFA0_deg*np.pi/180 # MFA at deposition
    MFA = np.arctan(np.tan(theta_depo)*np.exp(eps_t)) # micro-fibril angle evolution
    if (hasattr(p, "no_rotation") and p.no_rotation==True): 
        MFA = np.arctan(np.tan(theta_depo)*np.exp(np.zeros((nt,p.nl)))) # no rotation case
 
    # compute mean MFA angle : 
    for k in range(nt):
        MFA_mean[k] = np.dot(MFA[k],Wa[k])/WaT[k]
        MFA_median[k] = np.median(MFA[k])
    print(' ---- Mean MFA at the end is  ----')
    print(f"{MFA_mean[-1]/np.pi*180} deg")
    
    print(' ---- MEDIAN MFA at the end is  ----')
    print(f"{MFA_median[-1]/np.pi*180} deg")
    
    ### find the z position where the stress is the closest to the mean value
    idx_mean = (np.abs(sa[-1,:] - sa_mean[-1,0])).argmin()
    
    print(' ---- MFA at mean stress in the end is  ----')
    print(f"{MFA[-1,idx_mean]*180/np.pi} deg")
    
    # # Compute eigenvectors
    # # First, based on local layer values
    # Rz = np.sqrt(((sa-sl)/2)**2+tau**2)
    # phi1 = np.arctan(tau/(Rz+(sa-sl)/2))*180/np.pi
    
    # then based on averaged value
    Rz_mean = np.sqrt(((sa_mean-sl_mean)/2)**2+tau_mean**2)
    phi1_mean = 90 - np.arctan(tau_mean/(Rz_mean+(sa_mean-sl_mean)/2))*180/np.pi
    
    ###############################################
    if p.nl==1:
        ########## Find where Tsai-Hill got activated #######
        found_yield=False
        for k in range(nt):
            rotation = rotation_matrix_stress(np.pi/2-MFA[k,:]) #rotation matrix to get stress in the MF frame
            Contra_rotate = rotation_matrix_stress(MFA[k,:] - np.pi/2)
            s_rotated = np.zeros((3, len(MFA[k,:])))    
            for m in range(len(MFA[k,:])):
                # Perform multiplication for each layer -> IT IS NOT VECTORIZED !!! =(
                s_rotated [:, m] = rotation[m, :, :]  @ np.array((sa[k,m],sl[k,m],tau[k,m]))
                # compute Tsai-Hill criterion
                Tsai[:,k]= (s_rotated[0,:]/p.Y1)**2 - (s_rotated[0,:]*s_rotated[1,:])/(p.Y1**2) + (s_rotated[1,:]/p.Y2)**2 + (s_rotated[2,:]/p.Y12)**2
                plasticity = (Tsai[:,k]>1) # True if the wall layer is in plasticity
                buffer = compute_thresholds(p.radii, p.angle, s_rotated[0,:], s_rotated[1,:], s_rotated[2,:], plasticity)         
                yields_rotated[:,:,k] = Contra_rotate[m, :, :]  @ np.transpose(buffer)
                if plasticity[0]==True:
                  if found_yield==False:
                      first_yield = yields_rotated[0,0,k] # save apparent yield stress
                      print(' ---- Plasticity starts at  ----')
                      print(f"{yields_rotated[0,0,k]/1e6} MPa")
                      found_yield=True   
        if found_yield==False:
            print("!!! Yield not found !!!")
        # at convergence, the yields is the last computed value
        sY[i] = yields_rotated[0,0,-1] # here it is for the first layer
        print(' ---- Actual yield is ----')
        print(f"sigma_Y ={sY[i]/1e6} MPa")
          
        ############ Compute the elastic modulus of the first layer ########
        if read_data_file==False:
            mask = sa_mean<=0.1*first_yield # isolate stresses below yield    
            y = sa_mean[mask]    
            x = eps_t[mask[:,0],0]
            # if not y:
            #     y = sa_mean[0:2]   
            #     x = eps_t[0:2,0]
            #if len(y)<2:
            # apply a linear regression model
            m_x = np.mean(x)
            m_y = np.mean(y)
            E_stat = np.sum((x-m_x)*(y-m_y))/np.sum((x-m_x)**2)
            # basic method
            E_basic= y[-1]/x[-1] # just use endpoints
            # more advanced linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(fit_intercept=False).fit(x.reshape((-1, 1)), y)
            E_skl = model.coef_
            epsY[i] = sY[i]/E_skl[0]
            print(' ---- Computed Young modulus ----')
            print(f"E_basic ={E_basic/1e6} MPa")
            print(f"E_skl = {E_skl/1e6} MPa")
            print(f"E_stat = {E_stat/1e6} MPa")
            E[i] = E_skl[0]
            print(f"eps_Y is = {epsY[i]}")
            
            
            ############ save the previous data
            if parametric_study==True and save_data_parametric==True:
                for z in range(len(parameter_name)):
                    param2save[i,z] = getattr(p, parameter_name[z])
            
    
        if read_data_file==True: # read some data file
            size = len(data2open)
            print("Data file opened: " + str(data2open))
            path = "../runs/" + data2open[0]
            with open(path, "rb") as file:
                data = pickle.load(file)
                
            param = data.values
            my_E = data.E
            my_sY = data.sY 
            # find the correct angle
            indices = np.where(param == p.MFA0_deg)[0]
            # sY[i] = my_sY[indices][0]
            E[i] = my_E[indices][0]
            print(f"Angle is ={p.MFA0_deg} deg")
            print(f"E is ={E[i]/1e6} MPa")
            print(f"sY is ={sY[i]/1e6} MPa")
            
            
        ######## compute the viscous time constant
        dsadt = np.diff(sa_mean[:,0])/p.dt
        dsadt = np.concatenate(([0], dsadt))
        tau_visc = np.zeros((nt))
        for k in range(nt):
            #tau_visc[k] = (sa_mean[k]-sY[i])/(E[i]*G[k] - dsadt[k]+1e-10)/3600 # viscous time constant in h
            #tau_visc[k] = (sa_mean[k,0]-sY[i])/(E[i]*G[k,0] +1e-14)/3600 
            tau_visc[k] = (sa_mean[k,0]-yields_rotated[0,0,k])/(E[i]*G[k,0] +1e-14)/3600 
    
    
        tau_visc_save[i] = tau_visc[-1] # save viscous time constant at convergence
        print(' ---- Computed viscous time constant ----')
        print(f"Viscous time is = {tau_visc_save[i]} [h]")
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
        
    #################
    # Compute pressure and growth rate using (adapted) Lockhart solution
    PM = p.Psi_src + p.Pi0 # motor power
    Py0 = 0.05e6*2*WaT/(p.Lp-2*WaT) # yield pressure
    phi_w = 1/(p.tau_visc*p.C_el[0][0])*(p.Lp-2*WaT)/(2*WaT) # no synthesis case
    #phi_w = 1/p.mu*(p.Lp)/(2*WaT) # synthesis case
    phi_a = 2*(La + p.Lp)/((La-2*p.Wp0)*(p.Lp-2*WaT))*p.kh
    alpha = phi_a/(phi_a+phi_w)
    P_lock = alpha*PM + (1-alpha)*Py0 # no synthesis case
    # P_lock = alpha*PM + (1-alpha)*Py0 + p.omega/p.rho_w/(phi_a+phi_w)*1/(1+p.Rg*p.T*p.Km/PI)# synthesis case
    G_lock = alpha*phi_w*(PM-Py0) #+ p.omega/p.rho_w*alpha*1/(1+p.Rg*p.T*p.Km/PI)
    
    ############## save final mean anticlinal stress ##############
    if multilayer_study==True:
        sigma_saved[i] = sa_mean[-1,0] # save final stress
        print(f"Final stress is {sigma_saved[i]/1e6} [MPa]")
        for z in range(len(parameter_name)):
            param2save[i,z] = getattr(p, parameter_name[z])
    
    
    ########## Skip plots if you want
    if skip_plot==True: continue
    ######### Plots ##########
    # plt.figure(1,figsize=myfigsize)
    # plt.plot(th,(La)*1000000,label=my_legend[i])
    # # plt.plot([150, 150], [0, 84],':b',linewidth=1.5)
    # # plt.plot([90, 90], [0, 42],color='orange',linewidth=1.5,linestyle=':')
    # # plt.xlim((0,100))
    # # plt.ylim((0,1000))
    # #plt.legend(loc='best',title=legend_title)
    # #plt.yscale('log')
    # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # plt.ylabel(r"\textbf{R [$\mu$m]}", fontsize=16)
    # plt.title(r"$\textbf{Cell radius}$", fontsize=16)
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    

    # ax.plot(th,sa_mean/1e6,label=my_legend[i])
    # ax.legend(loc='best',title=legend_title)
    # #plt.plot([0, p.t_end/3600], [p.sig_Y/1e6, p.sig_Y/1e6],':k',linewidth=1.5)
    # # plt.ylim((1,1.5))
    # # plt.xlim((20,60))
    # ax.set_xlabel(r"\textbf{t [h]}", fontsize=16)
    # ax.set_ylabel(r"\textbf{$\overline{\sigma_a}$ [MPa]}", fontsize=16)
    # # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # # plt.ylabel(r"\textbf{$\overline{\sigma_a}$ [MPa]}", fontsize=16)
    # ax.set_title(r"$\textbf{Mean wall stress}$", fontsize=16)
    # # # Set grid and minor ticks
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax.minorticks_on()
    # # # Use LaTeX for tick labels (optional)
    # ax.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # # ax.indicate_inset([800,2.6,200,0.2],axins, edgecolor="blue")
    # plt.tight_layout()
    # plt.show()
    
    # Create the inset plot
    # axins.plot(th,sa_mean/1e6,label=my_legend[i])
    # axins.set_xlim(800, 1000)  # Adjust x-axis limits for the inset
    # axins.set_ylim(2.6, 2.8)  # Adjust y-axis limits for the inset
    # ax.indicate_inset_zoom(axins, edgecolor="blue")
    # Plot data in the inset
    #axins.plot(th,sa_mean/1e6)
    #axins.plot(x, y2, label='cos(x)')
    # axins.set_xlim(800, 1000)  # Adjust x-axis limits for the inset
    # axins.set_ylim(2.5, 2.7)  # Adjust y-axis limits for the inset

    
    plt.figure(2,figsize=myfigsize)
    plt.plot(th,sa_mean/1e6,label=my_legend[i],linestyle=lines[i])
    plt.legend(loc='best',title=legend_title)
    # if p.nl==1:
    #     plt.legend(loc='best',title=legend_title)
    #plt.plot([0, p.t_end/3600], [p.sig_Y/1e6, p.sig_Y/1e6],':k',linewidth=1.5)
    # plt.ylim((1,1.5))
    # plt.xlim((20,60))
    plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    plt.ylabel(r"\textbf{$\overline{\sigma_a}$ [MPa]}", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    ax = plt.gca()
    plt.text(-0.15, 1.05, '($\t{c}$)', transform=ax.transAxes, fontsize=16,
           verticalalignment='top', horizontalalignment='left')
    
    ##save figure
    #plt.savefig('multi_stress_stationnary.png',format='png', dpi=500)
    
    # plt.figure(13)
    # #plt.plot(th,yields_rotated[0,0,:]/1e6,label=my_legend[i],marker='x')
    # plt.plot(sa/1e6,yields_rotated[0,0,:]/1e6,label=my_legend[i],marker='x')
    # plt.legend(loc='best',title=legend_title)
    # plt.plot([0, sa[-1,0]/1e6], [sY[i]/1e6, sY[i]/1e6],':k',linewidth=1.5)
    # # plt.ylim((1,1.5))
    # # plt.xlim((20,60))
    # plt.xlabel(r"\textbf{${\sigma_a}$ [MPa]}", fontsize=16)
    # plt.ylabel(r"\textbf{${\sigma_Y^{a}}$ [MPa]}", fontsize=16)
    # plt.title(r"$\textbf{Yield stress in the global frame}$", fontsize=16)
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
    # if p.nl==1:
    #     plt.figure(18,figsize=myfigsize)
    #     plt.plot(th,yields_rotated[0,0,:]/1e6,label=my_legend[i],marker='x')
    #     # plt.plot(sa/1e6,yields_rotated[0,0,:]/1e6,label=my_legend[i],marker='x')
    #     plt.legend(loc='best',title=legend_title)
    
    #     plt.plot([0, th[-1]], [sY[i]/1e6, sY[i]/1e6],':k',linewidth=1.5)
    #     # plt.ylim((1,1.5))
    #     # plt.xlim((20,60))
    #     plt.xlabel(r"\textbf{${t_h}$ [h]}", fontsize=16)
    #     plt.ylabel(r"\textbf{${\sigma_Y^{a}}$ [MPa]}", fontsize=16)
    #     plt.title(r"$\textbf{Yield stress in the global frame}$", fontsize=16)
    #     # Set grid and minor ticks
    #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #     plt.minorticks_on()
    #     # Use LaTeX for tick labels (optional)
    #     plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    #     plt.tight_layout()
    
    # # plt.figure(18)
    # # plt.plot(th,yields[1,0,:]/1e6,label=my_legend[i])
    # # # plt.legend(loc='best',title=r"$MFA_0$")
    # # #plt.plot([0, p.t_end/3600], [p.sig_Y/1e6, p.sig_Y/1e6],':k',linewidth=1.5)
    # # # plt.ylim((1,1.5))
    # # # plt.xlim((20,60))
    # # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # # plt.ylabel(r"\textbf{${\sigma_Y^{1}}$ [MPa]}", fontsize=16)
    # # plt.title(r"$\textbf{Yield stress 1 in the MF frame}$", fontsize=16)
    # # # Set grid and minor ticks
    # # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # # plt.minorticks_on()
    # # # Use LaTeX for tick labels (optional)
    # # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # # plt.tight_layout()
    
    # plt.figure(14,figsize=myfigsize)
    # plt.plot(th,sl/1e6,label=my_legend[i])
    # if p.nl==1:
    #     plt.legend(loc='best',title=legend_title)
    # #plt.plot([0, p.t_end/3600], [p.sig_Y/1e6, p.sig_Y/1e6],':k',linewidth=1.5)
    # # plt.ylim((1,1.5))
    # # plt.xlim((20,60))
    # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # plt.ylabel(r"\textbf{${\sigma_l}$ [MPa]}", fontsize=16)
    # plt.title(r"$\textbf{longitudinal wall stress}$", fontsize=16)
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
    # plt.figure(15,figsize=myfigsize)
    # plt.plot(th,tau/1e6,label=my_legend[i])
    # # phi2 = np.arctan(tau/(-Rz+(sl-sa)/2))*180/np.pi
    # # plt.plot(th,90-phi1,label=my_legend[i],marker='o',linestyle='none')
    # # plt.plot(th,90-phi1_mean,color='black',linestyle='--')
    # # plt.plot(th,phi1-90+MFA*180/np.pi,label=my_legend[i],linestyle='--')
    # # plt.plot(th,lam2,label=my_legend[i],linestyle='--')
    # if p.nl==1:
    #     plt.legend(loc='best',title=legend_title)
    # #plt.plot([0, p.t_end/3600], [p.sig_Y/1e6, p.sig_Y/1e6],':k',linewidth=1.5)
    # # plt.ylim((1,1.5))
    # # plt.xlim((20,60))
    # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # plt.ylabel(r"\textbf{${\tau}$ [MPa]}", fontsize=16)
    # plt.title(r"$\textbf{Shear}$", fontsize=16)
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
    # plt.figure(16,figsize=myfigsize)
    # plt.plot(th,Tsai[0,:],label=my_legend[i])
    # # phi2 = np.arctan(tau/(-Rz+(sl-sa)/2))*180/np.pi
    # # plt.plot(th,90-phi1,label=my_legend[i],marker='o',linestyle='none')
    # # plt.plot(th,90-phi1_mean,color='black',linestyle='--')
    # # plt.plot(th,phi1-90+MFA*180/np.pi,label=my_legend[i],linestyle='--')
    # # plt.plot(th,lam2,label=my_legend[i],linestyle='--')
    # plt.legend(loc='best',title=legend_title)
    # #plt.plot([0, p.t_end/3600], [p.sig_Y/1e6, p.sig_Y/1e6],':k',linewidth=1.5)
    # # plt.ylim((1,1.5))
    # # plt.xlim((20,60))
    # plt.xlabel(r"\textbf{$t_h$ [h]}", fontsize=16)
    # plt.ylabel(r"\textbf{${Tsai}$}", fontsize=16)
    # plt.title(r"$\textbf{Tsai-Hill value}$", fontsize=16)
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
        
    # plt.figure(3,figsize=myfigsize)
    # plt.plot(th,P/1e6,label=my_legend[i])    
    # plt.plot([0, p.t_end/3600], [PM/1e6, PM/1e6],':k',linewidth=1.5)
    # #plt.plot(th,P_lock/1e6,marker='v', markevery=8, linestyle='',color=my_color[i])
    # plt.legend(loc='best',title=legend_title)
    # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # plt.ylabel(r"\textbf{$P$ [MPa]}", fontsize=16)
    # plt.title(r"$\textbf{Turgor pressure}$", fontsize=16)
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
    # plt.figure(4,figsize=myfigsize)
    
    # plt.plot(th,WaT*1000000,label=my_legend[i])
    # plt.legend(loc='best',title=legend_title)
    # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # plt.ylabel(r"\textbf{Wa [$\mu$m]}", fontsize=16)
    # plt.title(r"$\textbf{Wall thickness}$", fontsize=16)
    # # Set grid and minor ticks
    # # plt.xlim((80,100))
    # # plt.ylim((3.5,4.5))
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
    # # plt.figure(5)
    # # plt.plot(th,PI/1e6,color=my_color[i])
    # # # plt.legend(loc='best')
    # # plt.xlabel('t [h]')
    # # plt.ylabel('PI [MPa]')
    
    # plt.figure(6,figsize=myfigsize)
    # # plt.scatter(eps_t[:,0]*100,sa_mean/1e6,c=th,cmap='viridis')
    # # plt.colorbar(label='Time [h]')
    # plt.plot(eps_t[:,0]*100,sa_mean/1e6,label=my_legend[i])
    # if p.nl==1:
    #     plt.plot([eps_t[0,0]*100, eps_t[-1,0]*100], [sY[i]/1e6, sY[i]/1e6],':k',linewidth=2)
    #     plt.plot(eps_t[:,0]*100,eps_t[:,0]*E[i]/1e6,'orange',linestyle='--',linewidth=2)
    #     #plt.legend(loc='best',title=legend_title)
    #     text_str = f"$\sigma_Y$ : {sY[0]/1e6:.2f} MPa"
    #     text2 = f"E = {E[i]/1e6:.2f} MPa"
    #     plt.text(8,4.5,text_str, fontsize=14)
    #     plt.text(0.45,1.5,text2, fontsize=14, rotation=63)
    #     ax = plt.gca()
    #     plt.text(-0.15, 1.05, '($\t{a}$)', transform=ax.transAxes, fontsize=16,
    #             verticalalignment='top', horizontalalignment='left')
    # plt.ylim((0,1.2*np.max(sa_mean/1e6)))
    # plt.xlim((0,15))
    # plt.xlabel(r"\textbf{$\varepsilon_T$ $[\%]$}", fontsize=16)
    # # plt.ylabel(r"\textbf{$\overline{\sigma_a}$ [MPa]}", fontsize=16)
    # plt.ylabel(r"\textbf{${\sigma_a}$ [MPa]}", fontsize=16)
    # #plt.title(r"$\textbf{Stress-deformation curve for $\phi_0$="+str(p.MFA0_deg)+" degrees}$", fontsize=16)
    

    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
    # # save figure
    # #plt.savefig('traction_test.png',format='png', dpi=500)
    
    
    # plt.figure(7,figsize=myfigsize)
    # plt.plot(th,theta_MT,label = 'MT angle')
    # plt.plot(th,phi1_mean,'--',label = 'Principal stress angle')
    # plt.legend(loc='best')
    # plt.xlabel(r"\textbf{$t_h$ $[h]$}", fontsize=16)
    # plt.ylabel(r"\textbf{$\theta$ [$\circ$]}", fontsize=16)
    # # plt.title(r"$\textbf{stress-defo}$", fontsize=16)
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
    
    # # plt.plot(th,sa_mean/1e6,label=my_legend[i],color=my_color[i])
    # # plt.legend(loc='best',title=legend_title)
    # # #plt.plot([0, p.t_end/3600], [p.sig_Y/1e6, p.sig_Y/1e6],':k',linewidth=1.5)
    # # # plt.ylim((1,1.5))
    # # # plt.xlim((20,60))
    # # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # # plt.ylabel(r"\textbf{$\overline{\sigma_a}$ [MPa]}", fontsize=16)
    # # plt.title(r"$\textbf{Mean anticlinal wall stress}$", fontsize=16)
    # # # Set grid and minor ticks
    # # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # # plt.minorticks_on()
    # # # Use LaTeX for tick labels (optional)
    # # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # # plt.tight_layout()
    
    # # if len(list_files)==1:
    # #     plt.figure(7)
    # #     plt.plot(th,sa/1e6,label='anticlinal')
    # #     plt.plot(th,sl/1e6,'--',label='longitudinal')
    # #     plt.plot(th,tau/1e6,':',label='shear')
    # #     # plt.legend(loc='best')
    # #     #plt.plot([0, p.t_end/3600], [1, 1],':k',linewidth=1.5)
    # #     plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # #     plt.ylabel(r"\textbf{$\sigma$ [MPa]}", fontsize=16)
    # #     plt.title(r"$\textbf{Stresses}$", fontsize=16)
    # #     # Set grid and minor ticks
    # #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # #     plt.minorticks_on()
    # #     # Use LaTeX for tick labels (optional)
    # #     plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # #     plt.tight_layout()
    
    # # plt.figure(16)
    # # plt.plot(th,yields[0,0,:]/1e6,label='anticlinal')
    # # plt.plot(th,yields[1,0,:]/1e6,'--',label='longitudinal')
    # # plt.plot(th,yields[2,0,:]/1e6,'--',label='shear')
    # # plt.legend(loc='best')
    # # #plt.plot(th,sl,'--')
    # # #plt.plot([0, p.t_end/3600], [1, 1],':k',linewidth=1.5)
    # # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # # plt.ylabel(r"\textbf{$\sigma_Y$ [MPa]}", fontsize=16)
    # # plt.title(r"$\textbf{Yields for all layers}$", fontsize=16)
    # # # Set grid and minor ticks
    # # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # # plt.minorticks_on()
    # # # Use LaTeX for tick labels (optional)
    # # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # # plt.tight_layout()
    
    # # plt.figure(8)
    # # plt.plot(th,Q,label=my_legend[i])
    # # # plt.ylim((0,1.1))
    # # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # # plt.ylabel(r"\textbf{$Q$ [m3/s]}", fontsize=16)
    # # plt.title(r"$\textbf{Flow rate}$", fontsize=16)
    # # # Set grid and minor ticks
    # # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # # plt.minorticks_on()
    # # # Use LaTeX for tick labels (optional)
    # # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # # plt.tight_layout()
    
        

    
    # plt.figure(10,figsize=myfigsize)
    # plt.plot(th,G*3600,label=my_legend[i])
    # # plt.ylim((0,1.1))
    # plt.legend(loc='best',title=legend_title)
    # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # plt.ylabel(r"\textbf{Growth rate [1/h]}", fontsize=16)
    # plt.title(r"$\textbf{Growth rate}$", fontsize=16)
    # plt.legend(loc='best')
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
    # # plt.figure(11)
    # # plt.plot(th,eps_t*100)
    # # # plt.ylim((0,1.1))
    # # plt.xlabel(r"\textbf{t [h]}", fontsize=16)
    # # plt.ylabel(r"\textbf{$\varepsilon_T$ $[\%]$}", fontsize=16)
    # # plt.title(r"$\textbf{Total deformation in \%}$", fontsize=16)
    # # # Set grid and minor ticks
    # # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # # plt.minorticks_on()
    # # # Use LaTeX for tick labels (optional)
    # # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # # plt.tight_layout()
    
    # # plt.figure(12)
    # # plt.plot(W_sum[-1,:]*1e6,MFA[-1,:]*180/np.pi)
    # # # plt.ylim((0,1.1))
    # # plt.xlabel(r"\textbf{W [µm]}", fontsize=16)
    # # plt.ylabel(r"\textbf{MFA in degrees}", fontsize=16)
    # # plt.title(r"$\textbf{MFA final distribution for $\phi_0$="+str(p.MFA0_deg)+" degrees}$", fontsize=16)
    # # # Set grid and minor ticks
    # # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # # plt.minorticks_on()
    # # # Use LaTeX for tick labels (optional)
    # # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # # plt.tight_layout()
    
    # # plt.figure(13)
    # # plt.hist(MFA[-1]*180/np.pi,bins=20,density=True)
    # # # plt.ylim((0,1.1))
    # # plt.xlabel(r"\textbf{MFA range}", fontsize=16)
    # # plt.ylabel(r"\textbf{PDF}", fontsize=16)
    # # plt.title(r"$\textbf{MFA final distribution}$", fontsize=16)
    # # # Set grid and minor ticks
    # # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # # plt.minorticks_on()
    # # # Use LaTeX for tick labels (optional)
    # # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # # plt.tight_layout()
    
    # if p.nl==1:
    #     plt.figure(17,figsize=myfigsize)
    #     # plt.plot(G*3600,sa_mean,label=my_legend[i],color=my_color[i])
    #     # plt.scatter((sa_mean-sY[i])/1e6,G*3600,c=th,cmap='viridis')
    #     #mask = sa_mean>sY[i]
    #     plt.plot(th,tau_visc,label=my_legend[i])
    #     plt.plot([th[0],th[-1]],[5, 5],':k',linewidth=1.5)
    #     # plt.xlim([150,400])
    #     plt.ylim([0,10])
    #     # plt.plot(th,G_lock*3600,marker='v', markevery=8, linestyle='',color=my_color[i])
    #     # plt.ylim((0,1.1))
    #     # plt.yscale('log')
    #     plt.legend(loc='best',title=legend_title)
    #     plt.xlabel(r"\textbf{$t_h$ [h]}", fontsize=16)
    #     plt.ylabel(r"\textbf{$\tau_{visq}$ [h]}", fontsize=16)
    #     plt.title(r"$\textbf{Viscous time}$", fontsize=16)
    #     # Set grid and minor ticks
    #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #     plt.minorticks_on()
    #     # Use LaTeX for tick labels (optional)
    #     plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    #     plt.tight_layout()
    
    
    # plt.figure(9,figsize=myfigsize)
    # fig, ax = plt.subplots()
    # # Spatio-temporal stress profile
    # # Define your desired colormap (e.g., 'viridis')
    # cmap = plt.cm.viridis
    
    # # Select equally distributed timesteps (adjust divisor for different numbers)
    # selected_timesteps = np.linspace(25, nt-1, 50, dtype=int)  # Use np.linspace for even distribution
    
    # # Create a color mapper object for the selected timesteps
    # norm = plt.Normalize(vmin=th[selected_timesteps[0]], vmax=th[selected_timesteps[-1]])
    # sm = ScalarMappable(cmap=cmap, norm=norm)
    
    # # Plot loop for each timestep
    # for k,timestep in enumerate(selected_timesteps):
    #     # Get color based on timestep
    #     color = sm.to_rgba(th[timestep])
        
    #     # Get data points where wa is not zero (boolean mask)
    #     mask = Wa[timestep] != 0
        
    #     # Filter W_sum and sa based on the mask
    #     W_sum_f = W_sum[timestep][mask]
    #     sa_f = sa[timestep][mask]
        
    #     # add left point for plot clarity
    #     testW = np.insert(W_sum_f,0,0)
    #     testS = np.insert(sa_f,0,sa_f[0])
                
    #     # # Local average for non-edge elements (vectorized)
    #     # W_sum_f[1:-1] = (W_sum_f[:-2] + W_sum_f[2:]) / 2
        
    #     # # plot at mean layer thickness values
    #     # if len(W_sum_f) > 1:  # Check for at least two elements
    #     #     W_sum_f[0] = (W_sum_f[0]) / 2  # Average first element
    #     #     W_sum_f[-1] = (W_sum_f[-1] + W_sum_f[-2]) / 2  # Average last element
    #     # else:
    #     #     W_sum_f[0] = (W_sum_f[0]) / 2 

    #     # Plot with color from colormap
    #     ax.plot(testW*1e6, testS*1e-6, color=color, label=f'Time: {th[timestep]}')
        
    # # Add colorbar
    # fig.colorbar(sm, label='Time [h]', ax=ax)  # Add colorbar to the axes    
    # #plt.plot(W_sum[-1]*1e6,sa[-1]/1e6,color=my_color[i])
    # # plt.ylim((0,1.1))
    # #plt.plot([p.Wa0/(1+p.sig_Y/p.E)*1e6, p.Wa0/(1+p.sig_Y/p.E)*1e6], [0, np.max(np.max(sa, axis=1))/1e6],':k',linewidth=1.5)
    # plt.xlabel(r"\textbf{Position within the wall [µm]}", fontsize=16)
    # plt.ylabel(r"\textbf{$\sigma_a$ [MPa]}", fontsize=16)
    # #plt.title(r"$\textbf{Wall stress profiles}$", fontsize=16)
    # #plt.legend(loc='best')
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    # dpi = 500  # Dots per inch
    # ax = plt.gca()
    # plt.text(-0.1, 1.05, '($\t{d}$)', transform=ax.transAxes, fontsize=16,
    #         verticalalignment='top', horizontalalignment='left')
    # # Save the figure as a high-resolution JPEG
    # # plt.savefig('wall_stress_non_stationnary_with_rotation.png',format='png', dpi=500)
    
    
       
    # plt.figure(13,figsize=myfigsize)
    # fig, ax = plt.subplots()
    # # Spatio-temporal stress profile
    # # Define your desired colormap (e.g., 'viridis')
    # cmap = plt.cm.viridis
    
    # # Select equally distributed timesteps (adjust divisor for different numbers)
    # selected_timesteps = np.linspace(25, nt-1, 10, dtype=int)  # Use np.linspace for even distribution
    
    # # Create a color mapper object for the selected timesteps
    # norm = plt.Normalize(vmin=th[selected_timesteps[0]], vmax=th[selected_timesteps[-1]])
    # sm = ScalarMappable(cmap=cmap, norm=norm)
    
    # # Plot loop for each timestep
    # for k,timestep in enumerate(selected_timesteps):
    #     # Get color based on timestep
    #     color = sm.to_rgba(th[timestep])
        
    #     # Get data points where wa is not zero (boolean mask)
    #     mask = Wa[timestep] != 0
        
    #     # Filter W_sum and sa based on the mask
    #     W_sum_f = W_sum[timestep][mask]
    #     MFA_f = MFA[timestep][mask]
        
    #     # add left point for plot clarity
    #     testW = np.insert(W_sum_f,0,0)
    #     testMFA= np.insert(MFA_f,0,MFA_f[0])
                
    #     # # Local average for non-edge elements (vectorized)
    #     # W_sum_f[1:-1] = (W_sum_f[:-2] + W_sum_f[2:]) / 2
        
    #     # # plot at mean layer thickness values
    #     # if len(W_sum_f) > 1:  # Check for at least two elements
    #     #     W_sum_f[0] = (W_sum_f[0]) / 2  # Average first element
    #     #     W_sum_f[-1] = (W_sum_f[-1] + W_sum_f[-2]) / 2  # Average last element
    #     # else:
    #     #     W_sum_f[0] = (W_sum_f[0]) / 2 

    #     # Plot with color from colormap
    #     ax.plot(testW*1e6, testMFA*180/np.pi, color=color, label=f'Time: {th[timestep]}')
        
        
        
    # # Add colorbar
    # fig.colorbar(sm, label='Time [h]', ax=ax)  # Add colorbar to the axes    
    # #plt.plot(W_sum[-1]*1e6,sa[-1]/1e6,color=my_color[i])
    # # plt.ylim((0,1.1))
    # #plt.plot([p.Wa0/(1+p.sig_Y/p.E)*1e6, p.Wa0/(1+p.sig_Y/p.E)*1e6], [0, np.max(np.max(sa, axis=1))/1e6],':k',linewidth=1.5)
    # plt.xlabel(r"\textbf{Position within the wall [µm]}", fontsize=16)
    # plt.ylabel(r"\textbf{$\phi^i$ [$\circ$]}", fontsize=16)
    # #plt.title(r"$\textbf{MFA profiles}$", fontsize=16)
    # #plt.legend(loc='best')
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    # dpi = 300  # Dots per inch
    # # Save the figure as a high-resolution JPEG
    # #plt.savefig("stress_profiles-W_cste.jpeg", dpi=dpi)
    # ax = plt.gca()
    # plt.show
    # plt.text(-0.15, 1.05, '($\t{c}$)', transform=ax.transAxes, fontsize=16,
    #         verticalalignment='top', horizontalalignment='left')
    # # Save the figure as a high-resolution JPEG
    # #.savefig('MFA_non_stationnary_with_rotation.png',format='png', dpi=500)
    
    #########################   Kimograph MFA #####################

    # Utiliser pcolormesh pour tracer les données en fonction des positions réelles
    # Créer une grille 2D pour les positions et les temps
    # Ajouter une colonne fictive pour la couche 0
    W_sum_with_fictive = np.zeros((nt, p.nl + 1))
    W_sum_with_fictive[:, 1:] = W_sum
    
    MFA_with_fictive = np.zeros((nt, p.nl + 1))
    MFA_with_fictive[:, 1:] = MFA
    MFA_with_fictive[:, 0] = MFA[:, 0]  # La couche fictive a la même valeur que la première couche

    X, Y = np.meshgrid(np.linspace(0, 1, p.nl+1), th)
    
    ###Utiliser W_sum pour ajuster les positions des couches
    for k in range(nt):
        X[k, :] = W_sum_with_fictive[k, :]
        #X[k, :] = W_sum[k, :]
    
    plt.figure(14,figsize=(15/2.54, 10/2.54))
    c=plt.pcolormesh(X*1e6, Y, MFA_with_fictive * 180 / np.pi, shading='auto', cmap='inferno_r',edgecolor='none')
    plt.xlim([0,1])
    colorbar = plt.colorbar(c)
    colorbar.set_label(label=r'\textbf{$\phi^i$ [$\circ$]}', size=14)  # Augmenter la taille de la police
    plt.xlabel(r"\textbf{Position within the wall [µm]}", fontsize=16)
    plt.ylabel(r"\textbf{$t$ [h]}", fontsize=16)
    
    
    # Ajouter un insert pour montrer la courbe MFA en fonction de W_sum pour le dernier pas de temps
    ax_insert = inset_axes(plt.gca(), width="40%", height="40%", loc='lower right', borderpad=2)
    ax_insert.plot(W_sum_with_fictive[-1]*1e6, MFA_with_fictive[-1] * 180 / np.pi, label='Final time')
    #ax_insert.set_xlabel('W_sum')
    ax_insert.set_ylabel(r'\textbf{$\phi^i$ [$\circ$]}')
    # ax_insert.legend()
    
    # Rendre l'insert transparent et ajouter une grille
    ax_insert.patch.set_alpha(0.5)  # Rendre le fond transparent
    ax_insert.grid(True)  # Ajouter une grille
    #plt.savefig('MFA_stationnary.png',format='png', dpi=500) 
    # plt.text(-0.1, 0.91, '($\t{a}$)', transform=ax.transAxes, fontsize=16,
    #          verticalalignment='top', horizontalalignment='left')
    # plt.savefig('MFA_non_stationnary_with_rotation.png',format='png', bbox_inches='tight', dpi=500) 
    plt.show()
    
    #
    
    #########################   Kimograph stress #####################

    # Utiliser pcolormesh pour tracer les données en fonction des positions réelles
    # Créer une grille 2D pour les positions et les temps
    # Ajouter une colonne fictive pour la couche 0
    
    sa_with_fictive = np.zeros((nt, p.nl + 1))
    sa_with_fictive[:, 1:] = sa
    sa_with_fictive[:, 0] = sa[:, 0]  # La couche fictive a la même valeur que la première couche
   
    plt.figure(15,figsize=(15/2.54, 10/2.54))
    c=plt.pcolormesh(X*1e6, Y, sa_with_fictive/1e6, shading='auto', cmap='inferno_r',edgecolor='none')
    plt.xlim([0,1])
    colorbar = plt.colorbar(c)
    colorbar.set_label(label=r'\textbf{$\sigma_a$ [MPa]}', size=14)  # Augmenter la taille de la police
    plt.xlabel(r"\textbf{Position within the wall [µm]}", fontsize=16)
    plt.ylabel(r"\textbf{$t$ [h]}", fontsize=16)
    
    
    # Ajouter un insert pour montrer la courbe MFA en fonction de W_sum pour le dernier pas de temps
    ax_insert = inset_axes(plt.gca(), width="40%", height="40%", loc='lower right', borderpad=2)
    ax_insert.plot(W_sum_with_fictive[-1]*1e6, sa_with_fictive[-1]/1e6, label='Final time')
    #ax_insert.set_xlabel('W_sum')
    ax_insert.set_ylabel(r'\textbf{$\sigma_a$ [MPa]}')
    # ax_insert.legend()
    
    # Rendre l'insert transparent et ajouter une grille
    ax_insert.patch.set_alpha(0.5)  # Rendre le fond transparent
    ax_insert.grid(True)  # Ajouter une grille
    #plt.savefig('wall_stress_stationnary.png',format='png', dpi=500)
    # plt.text(-0.1, 0.91, '($\t{b}$)', transform=ax.transAxes, fontsize=16,
    #          verticalalignment='top', horizontalalignment='left')
    # plt.savefig('wall_stress_non_stationnary_with_rotation.png',format='png', bbox_inches='tight',dpi=500)
    #

## write some data in a .txt file    
if parametric_study==True and save_data_parametric==True and multilayer_study==False:
    buff = data2write(parameter_name, param2save, sY, E, epsY, tau_visc_save)
    with open("../runs/" + data_file, 'wb') as file:
        pickle.dump(buff, file) # save data
        
## write some data in a .txt file    
if parametric_study==True and save_data_parametric==True and multilayer_study==True:
    buff = data2write_multi(parameter_name, param2save, sigma_saved)
    with open("../runs/" + data_file, 'wb') as file:
        pickle.dump(buff, file) # save data
  

    