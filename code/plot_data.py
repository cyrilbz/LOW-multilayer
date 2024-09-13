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
data2open = ['effect_AR.pkl']
data2open = ['effect_eps0.pkl']
# my_legend = ['30deg ','60','90deg']
param2change = 'eps0' # as written here below in the ref_values class
legend_title = r"$\varepsilon_0$"
legend_scaling = 1 # rescale legend values for easier reading
xtitle = r"\textbf{MFA [°]}"

treat_multilayer_case = True
if treat_multilayer_case==True:
    multi_file2open = 'multi-effect_eps0.pkl'

# define some linestyle cycling
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
# define some markers cycling
markers = ["x","v"]
markercycler = cycle(markers)
# define the color map
cmap = plt.cm.cool


# function for reference values computations
class ref_values:
    def __init__(self):

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
        
        # Yield deformation
        self.eps0 = 0.03
        
        # initialize some parameters that are computed later
        self.C_el = None
        self.nu12 = None
        
        self.Y1 = None
        self.Y2 = None
        self.Y12 = None
        
        self.Q11 = None
        self.Q22 = None
        self.Q12 = None
        self.Q66 = None
             
        # compute the previous parameters
        self.compute_dependencies()
        
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
        self.nu12 = self.num*(1+self.xi*eta_nu*self.Vf)/(1-eta_nu*self.Vf)
        nu21 = self.nu12*E2/E1
        
        # compute elastic matrix coefiicient in plane stress conditions:
        self.Q11 = E1/(1-self.nu12*nu21)
        self.Q22 = E2/(1-self.nu12*nu21)
        self.Q12 = nu21*self.Q11
        self.Q66 = G12
        
        # Tsai-hill criterion parameters
        self.Y1 = self.Q11*self.eps0
        self.Y2 = self.Q22*self.eps0
        self.Y12 = self.Q66*self.eps0
        
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
        Q11 = np.zeros((nv2))
        Q22 = np.zeros((nv2))
        Y1 = np.zeros((nv2))
        Y2 = np.zeros((nv2))
        Y12 = np.zeros((nv2))
        nu12 = np.zeros((nv2))
        
        # compute reference values for mechanical parameters
        ref = ref_values()
        
        for v2 in range(nv2):
            ref.set_param(param2change,value_2nd[v2]) # update ref values
            Q11[v2] = ref.Q11 # save Q11
            Q22[v2] = ref.Q22 # save Q22
            Y1[v2] = ref.Y1 # save Y1
            Y2[v2] = ref.Y2 # save Y2
            Y12[v2] = ref.Y12
            nu12[v2] = ref.nu12 # save  nu12
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
        
        
    ################### Treat the multilayer data #######################
    if treat_multilayer_case==True:
        print("Data file opened: " + str(multi_file2open))
        path = "../runs/" + multi_file2open
        with open(path, "rb") as file:
            data_multi = pickle.load(file)
        param_multi = data_multi.values 
        buff_stress = data_multi.stress
        
        # sort and order results
        value_param = np.unique(param_multi[:,1]) # take the parameter values and sort them
        value_G = np.unique(param_multi[:,0]) # take the G values and sort them
        nv_multi = len(value_param) # number of different parameter values
        sY_multi = np.zeros((nv_multi)) # to store the Yield values
        E_multi = np.zeros((nv_multi))
        tau_visc = 5*3600 # viscous time constant
        for z in range(nv_multi): # loop over G values and save corresponding stress
            it = param_multi[:,1]==value_param[z]
            stress_one = buff_stress[it][0]
            stress_two = buff_stress[it][1]
            G_one = param_multi[it,0][0]
            G_two = param_multi[it,0][1]
            sY_multi[z] = (stress_one-stress_two*G_one/G_two)/(1-G_one/G_two)
            E_multi[z] = (stress_one - sY_multi[z])/(tau_visc*G_one)
            
    ######################## Plot stuff ###########################
    # Normalize colors between 0 and 1 for even distribution across plots
    norm = plt.Normalize(vmin=0, vmax=len(E[0])-1)
    
    plt.figure(1)
    for k in range(nv2):
        color = cmap(norm(k))  # Get color based on normalized index
        value = np.sqrt(100)*Y2[k]*Y1[k]/np.sqrt(100*Y1[k]**2-9*Y2[k]**2)
        value = Y1[k]
        plt.plot(value_first,sY[:,k]/1e6, color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))
        plt.plot([0, value_first[-1]], [sY_multi[k]/1e6,sY_multi[k]/1e6],color = color,linewidth=1.5,ls='--')
        
    plt.legend(value_2nd*legend_scaling,loc='best',title=legend_title)
    # plt.plot([0, value_first[-1]], [ref.Y1/1e6,ref.Y1/1e6],':k',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [ref.nu12*ref.Y2/1e6, ref.nu12*ref.Y2/1e6],':r',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [2.78, 2.78],':g',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [0.98, 0.98],':b',linewidth=1.5)
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
        plt.plot([0, value_first[-1]], [E_multi[k]/1e6,E_multi[k]/1e6],color = color,linewidth=1.5,ls='--')
        #plt.plot([0, value_first[-1]], [Q11[k]/1e6,Q11[k]/1e6],color = color,linewidth=1.5,ls='--')
    plt.legend(value_2nd*legend_scaling,loc='best',title=legend_title)
    # plt.plot([0, value_first[-1]], [Q11, Q11],':k',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [Q22, Q22],':r',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [83.3, 83.3],':g',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [33.3, 33.3],':b',linewidth=1.5)
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
    
    plt.figure(6)
    for k in range(nv2):
        color = cmap(norm(k))  # Get color based on normalized index
        plt.plot(value_first,(E[:,k]-Q22[k])/(Q11[k]-Q22[k]), color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))
        # plt.plot([0, value_first[-1]], [Q11[k]/1e6,Q11[k]/1e6],color = color,linewidth=1.5,ls='--')
    plt.legend(value_2nd*legend_scaling,loc='best',title=legend_title)
    # plt.plot([0, value_first[-1]], [Q11, Q11],':k',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [Q22, Q22],':r',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [83.3, 83.3],':g',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [33.3, 33.3],':b',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [177.4, 177.4],':k',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [12.8, 12.8],':k',linewidth=1.5)
    #plt.yscale('log')
    plt.xlabel(xtitle, fontsize=16)
    plt.ylabel(r"\textbf{$(E-Q22)/(Q11-Q22)$}", fontsize=16)
    plt.title(r"\textbf{MFA effect on Young's modulus}", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    plt.figure(7)
    for k in range(nv2):
        color = cmap(norm(k))  # Get color based on normalized index
        plt.plot(value_first,(sY[:,k]-Y2[k])/(Y1[k]-Y2[k]), color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))

    plt.legend(value_2nd*legend_scaling,loc='best',title=legend_title)
    # plt.plot([0, value_first[-1]], [ref.Y1/1e6,ref.Y1/1e6],':k',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [ref.nu12*ref.Y2/1e6, ref.nu12*ref.Y2/1e6],':r',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [2.78, 2.78],':g',linewidth=1.5)
    # plt.plot([0, value_first[-1]], [0.98, 0.98],':b',linewidth=1.5)
    #plt.yscale('log')
    plt.xlabel(xtitle, fontsize=16)
    plt.ylabel(r"\textbf{$(\sigma_Y-Y_2)/(Y_1-Y_2)$}", fontsize=16)
    plt.title(r"\textbf{MFA effect on yield stress}", fontsize=16)
    # Set grid and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # Use LaTeX for tick labels (optional)
    plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    # plt.figure(3)
    # for k in range(nv2):
    #     color = cmap(norm(k))  # Get color based on normalized index
    #     plt.plot(value_first,epsY[:,k], color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))
    # plt.legend(value_2nd*legend_scaling,loc='best',title=legend_title)
    # #plt.plot([0, param[-1]], [0.03, 0.03],':k',linewidth=1.5)
    # # plt.plot([0, param[-1]], [177.4, 177.4],':k',linewidth=1.5)
    # # plt.plot([0, param[-1]], [12.8, 12.8],':k',linewidth=1.5)
    # #plt.yscale('log')
    # plt.xlabel(r"\textbf{MFA [°]}", fontsize=16)
    # plt.ylabel(r"\textbf{$\varepsilon_Y$ [-]}", fontsize=16)
    # plt.title(r"\textbf{MFA effect on threshold deformation}", fontsize=16)
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
    # if hasattr(data,"tau_visc")==True: 
    #     plt.figure(4)
    #     for k in range(nv2):
    #         color = cmap(norm(k))  # Get color based on normalized index
    #         plt.plot(value_first,tau_visc[:,k], color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))
    #     plt.legend(value_2nd*legend_scaling,loc='best',title=legend_title)
    #     # plt.plot([0, param[-1]], [5, 5],':k',linewidth=1.5)
    #     # plt.plot([0, param[-1]], [177.4, 177.4],':k',linewidth=1.5)
    #     # plt.plot([0, param[-1]], [12.8, 12.8],':k',linewidth=1.5)
    #     #plt.yscale('log')
    #     plt.xlabel(r"\textbf{MFA [°]}", fontsize=16)
    #     plt.ylabel(r"\textbf{$tau_{visc}$ [-]}", fontsize=16)
    #     plt.title(r"\textbf{MFA effect on viscous time constant}", fontsize=16)
    #     # Set grid and minor ticks
    #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #     plt.minorticks_on()
    #     # Use LaTeX for tick labels (optional)
    #     plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    #     plt.tight_layout()
        
    
    # plt.figure(5)
    # for k in range(nv2):
    #     color = cmap(norm(k))  # Get color based on normalized index
    #     plt.plot(value_first,2*sY[:,k]*1e-6/(2e-5)/1e6, color = color, linewidth=2,marker=next(markercycler),ls=next(linecycler))
    # plt.legend(value_2nd*legend_scaling,loc='best',title=legend_title)
    # # plt.plot([0, value_first[-1]], [177.4, 177.4],':k',linewidth=1.5)
    # # plt.plot([0, value_first[-1]], [12.8, 12.8],':k',linewidth=1.5)
    # #plt.yscale('log')
    # plt.xlabel(xtitle, fontsize=16)
    # plt.ylabel(r"\textbf{$P_Y$ [MPa]}", fontsize=16)
    # plt.title(r"\textbf{MFA effect on yield pressure}", fontsize=16)
    # # Set grid and minor ticks
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.minorticks_on()
    # # Use LaTeX for tick labels (optional)
    # plt.tick_params(labelsize=12, which='both', top=True, bottom=True, left=True, right=True)
    # plt.tight_layout()
    
################ perform the fits ############################

print("Now I perform the fits")
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

x = value_first # save angle
y = (E[:,2]-Q22[2])/(Q11[2]-Q22[2])# save young modulus
y = (sY[:,2]-Y2[2])/(Y1[2]-Y2[2])

# Define the logistic function
def logistic_func(x, L, k, x0):
    return L/ (1 + np.exp(-k * (x - x0)))

# # Define the logistic function
# def logistic_func(x, x0):
#     return 1.01/ (1 + np.exp(-0.09* (x - x0)))

# inital guess
initial_guess = [1,0.09,55]
#initial_guess = [55]

# Fit the logistic curve to the data
popt, pcov = curve_fit(logistic_func, x, y, p0=initial_guess)

# Extract the fitted parameters
L_fit, k_fit, x0_fit = popt
#x0_fit = popt

# Generate the fitted curve
y_fit = logistic_func(x, L_fit,  k_fit, x0_fit)
#y_fit = logistic_func(x, x0_fit)
# y_fit = logistic_func(x, 1,  0.12, 55)

# Calculate the Pearson correlation coefficient
correlation, p_value = pearsonr(y, y_fit)

print("Pearson Correlation Coefficient:", correlation)

# Assuming you have the fitted parameters L_fit, k_fit, and x0_fit

# Create the equation string with placeholders for the parameters
equation_str = r"$y = \frac{{L}}{{1 + e^{-{k}(x - {x_0})}}}$"
parameters = f" L={L_fit:.3f}, k={k_fit:.3f}, $x_0$={x0_fit:.3f}"
#parameters = f" x_0={x0_fit[0]:.3f}"
reg = f"$R^2$={correlation:.4f}"
# # Replace the placeholders with the actual values
# equation_text = equation_str.format(L=L_fit, k=k_fit, x0=x0_fit)

# Plot the data and the fitted curve
plt.figure(8)
plt.plot(x, y, 'bo', label='Data')
plt.plot(x, y_fit, 'r-', label='Fit')
plt.xlabel('Angle')
# Add the equation to the plot
plt.text(20, 0.8, equation_str, fontsize=12,color='red')
plt.text(5, 0.6, parameters, fontsize=12)
plt.text(10, 0.4, reg, fontsize=12)
plt.ylabel('Normalized Mechanical Property')
plt.legend()
plt.show()