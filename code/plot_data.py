# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:48:30 2024

@author: cbozonnet
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from itertools import cycle
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# data2open = ["data_effect_MFA-lowdt.pkl"]
# data2open = ['data_effect_MFA-forced_G.pkl']
data2open = ['data_effect_MFA-fine.pkl','data_effect_MFA-fine-no_fish.pkl']
# data2open = ['data_effect_MFA-fine.pkl']
data2open = ['data-test_parametric.pkl']
# my_legend = ['30deg ','60','90deg']
legend_title = r"$V_f$"
xtitle = r"\textbf{MFA [°]}"

# define some linestyle cycling
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
# define some markers cycling
markers = ["x","v"]
markercycler = cycle(markers)
# define the color map
cmap = plt.cm.cool



size = len(data2open)
for i in range(size): # loop to open the files one by one and plot things
    print("Data file opened: " + str(data2open[i]))
    path = "../runs/" + data2open[i]
    with open(path, "rb") as file:
        data = pickle.load(file)
        
    # load the save data
    param = data.values 
    buff_E = data.E
    buff_sY = data.sY 
    buff_epsY= data.epsY
    if hasattr(data,"tau_visc")==True: buff_tau_visc = data.tau_visc
    
    # sort and order them  in easily plottable data
    num_param = len(param[0]) # find the number of parameters
    if num_param==2:
        value_2nd = np.unique(param[:,1]) # take the values and sort them
        value_first = np.unique(param[:,0])
        nv2 = len(value_2nd) # number of different values for 2nd parameter
        nv1 = len(value_first)
        E = np.zeros((nv1,nv2))
        sY = np.zeros((nv1,nv2))
        epsY = np.zeros((nv1,nv2))
        tau_visc = np.zeros((nv1,nv2))
        for v2 in range(nv2):
            for v1 in range(nv1):
                it1 = param[:,0]==value_first[v1]
                it2 = param[:,1]==value_2nd[v2]
                E[v1,v2] = buff_E[it1*it2][0]
                sY[v1,v2] = buff_sY[it1*it2][0]
                epsY[v1,v2] = buff_epsY[it1*it2][0]
                tau_visc[v1,v2] = buff_tau_visc[it1*it2][0]
    elif num_param==1:
        value = param[:,0]
    else:
        raise Exception("Parameter number not handled currently")
    
    # Normalize colors between 0 and 1 for even distribution across plots
    norm = plt.Normalize(vmin=0, vmax=num_param - 1)
    
    plt.figure(1)
    for k in range(nv2):
        color = cmap(norm(k))  # Get color based on normalized index
        plt.plot(value_first,sY[:,k]/1e6, color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))
    plt.legend(value_2nd,loc='best',title=legend_title)
    # plt.plot([0, value_first[-1]], [177.4, 177.4],':k',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [12.8, 12.8],':k',linewidth=1.5)
    #plt.yscale('log')
    plt.xlabel(xtitle, fontsize=16)
    plt.ylabel(r"\textbf{$\sigma_Y$ [MPa]}", fontsize=16)
    plt.title(r"\textbf{MFA effect on yield stress}", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
        
    plt.figure(2)
    for k in range(nv2):
        color = cmap(norm(k))  # Get color based on normalized index
        plt.plot(value_first,E[:,k]/1e6, color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))
    plt.legend(value_2nd,loc='best',title=legend_title)
    # plt.plot([0, value_first[-1]], [177.4, 177.4],':k',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [12.8, 12.8],':k',linewidth=1.5)
    #plt.yscale('log')
    plt.xlabel(xtitle, fontsize=16)
    plt.ylabel(r"\textbf{$E$ [MPa]}", fontsize=16)
    plt.title(r"\textbf{MFA effect on Young's modulus}", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(3)
    for k in range(nv2):
        color = cmap(norm(k))  # Get color based on normalized index
        plt.plot(value_first,epsY[:,k], color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))
    plt.legend(value_2nd,loc='best',title=legend_title)
    #plt.plot([0, param[-1]], [0.03, 0.03],':k',linewidth=1.5)
    # plt.plot([0, param[-1]], [177.4, 177.4],':k',linewidth=1.5)
    # plt.plot([0, param[-1]], [12.8, 12.8],':k',linewidth=1.5)
    #plt.yscale('log')
    plt.xlabel(r"\textbf{MFA [°]}", fontsize=16)
    plt.ylabel(r"\textbf{$\varepsilon_Y$ [-]}", fontsize=16)
    plt.title(r"\textbf{MFA effect on threshold deformation}", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    if hasattr(data,"tau_visc")==True: 
        plt.figure(4)
        for k in range(nv2):
            color = cmap(norm(k))  # Get color based on normalized index
            plt.plot(value_first,tau_visc[:,k], color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))
        plt.legend(value_2nd,loc='best',title=legend_title)
        # plt.plot([0, param[-1]], [5, 5],':k',linewidth=1.5)
        # plt.plot([0, param[-1]], [177.4, 177.4],':k',linewidth=1.5)
        # plt.plot([0, param[-1]], [12.8, 12.8],':k',linewidth=1.5)
        #plt.yscale('log')
        plt.xlabel(r"\textbf{MFA [°]}", fontsize=16)
        plt.ylabel(r"\textbf{$tau_{visc}$ [-]}", fontsize=16)
        plt.title(r"\textbf{MFA effect on viscous time constant}", fontsize=16)
        # Set grid and minor ticks
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        # Use LaTeX for tick labels (optional)
        plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
        plt.tight_layout()
        