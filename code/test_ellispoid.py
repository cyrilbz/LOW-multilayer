# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:28:38 2024

@author: cbozonnet
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from functions import compute_thresholds
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

########### compute ellipsoid radii ##############

epsY = 0.03 # threshold deformation
# elastic constants
Q11 = 177e6
Q22 = 12.8e6
Q12 = 4.91e6
# radii in the original frame
Y1 = Q11*epsY
Y2 = Q22*epsY
Y12 = Q12*epsY*1000
# perform change of variable
alpha = 0.5*np.arctan(Y2**2/(Y2**2-Y1**2)) # angle of rotation
# here below compute the radii
a_1 = np.sqrt(1/(cos(alpha)**2/Y1**2 + sin(alpha)**2/Y2**2 +cos(alpha)*sin(alpha)/Y1**2))
a_2 = np.sqrt(1/(sin(alpha)**2/Y1**2 + cos(alpha)**2/Y2**2 -cos(alpha)*sin(alpha)/Y1**2))
a_3 = Y12

############### compute coordinates ################

dphi, dtheta = pi/10.0, pi/10.0 # angular step
[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta] # angular grid

xyz = np.zeros((np.size(phi), 3)) # data storage

center = [0, 0, 0] #ellipse center coordinates
radii = [a_1, a_2, a_3] # ellipse coefficient : [a_1,a_2,a_3]
# Compute variables using the parameteric equation
x = center[0] + radii[0] * np.cos(theta) * np.sin(phi)
w = center[1] + radii[1] * np.sin(theta) * np.sin(phi)
z = center[2] + radii[2] * np.cos(phi)
xyz[:, 0] = np.reshape(x, -1)
xyz[:, 1] = np.reshape(w, -1)
xyz[:, 2] = np.reshape(z, -1)

################## Compute minimal distance between the ellispe and a given point #####


s = [8e6,-1e6,6e6]
toto=np.zeros_like(s)

def find_location(coefs,point): 
    # a function that finds the closest location with a given point on an ellipsoid 
    # coefs are for the ellipsoid (the ellipsoid radii), point for the outside point
    # this is solved through Lagrange multiplier methodology
    lam = 0 # inital value for Lagrange multiplier
    n=0 # number of Newton iteration
    n_max = 60 # max number of iteration
    res = 1e-6 # residual
    glam = 10*res # intial value to enter the loop
    while abs(glam)>res and n<n_max: # Newton algorithm to find lambda
        # compute g(lambda) function (objective function to nullify)
        glam = (point[0]/(coefs[0]*(1-lam/coefs[0]**2)))**2 \
            + (point[1]/(coefs[1]*(1-lam/coefs[1]**2)))**2 \
            + (point[2]/(coefs[2]*(1-lam/coefs[2]**2)))**2 -1  
        # compute dg/dlmabda 
        dglam = 2*point[0]**2/(coefs[0]**4*(1-lam/coefs[0]**2)**3) \
            + 2*point[1]**2/(coefs[1]**4*(1-lam/coefs[1]**2)**3) \
            + 2*point[2]**2/(coefs[2]**4*(1-lam/coefs[2]**2)**3)             
        lam = lam - glam/dglam # Newton iteration
        n=n+1
    if (n>=n_max): 
        raise Exception("Projection onto the ellipsoid did not work")
    # compute solution
    x1 = point[0]/(1-lam/coefs[0]**2)
    x2 = point[1]/(1-lam/coefs[1]**2)
    x3 = point[2]/(1-lam/coefs[2]**2)
    return [x1,x2,x3]

y1 = np.cos(alpha)*s[0] - np.sin(alpha)*s[1]
y2 = np.sin(alpha)*s[0] + np.cos(alpha)*s[1]
point = [y1,y2,s[2]]   

toto = find_location(radii, point)
beta = np.zeros_like(toto)
beta[0]= cos(alpha)*toto[0] + sin(alpha)*toto[1]
beta[1] = - sin(alpha)*toto[0] + cos(alpha)*toto[1]
beta[2] = toto[2]  

print("beta1  | beta2 | beta3")
print("------|--------|------")
print("First algorithm")
print(f"{beta[0]:.6f} | {beta[1]:.6f} | {beta[2]:.6f}")

res = compute_thresholds(radii, alpha, [s[0]], [s[1]], [s[2]], [True])
print("Function")
print(res)
############### plots below ###########################
# from mayavi import mlab
# # axes = mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
# # mlab.points3d(x, y, z, color=(0, 1, 0))  # Green color
# s = mlab.mesh(x, y, z, opacity=1)

# mlab.show()
plt.ion()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x/1e6, w/1e6, z/1e6, rstride=1, cstride=1, cmap="plasma",linewidth=0, antialiased=False)
# ax.scatter(x/1e6, y/1e6, z/1e6)
ax.scatter(s[0]/1e6,s[1]/1e6,s[2]/1e6, marker='s', color='red')
ax.scatter(toto[0]/1e6,toto[1]/1e6,toto[2]/1e6, marker='^', color='red')
ax.plot([toto[0]/1e6,s[0]/1e6],[toto[1]/1e6,s[1]/1e6],[toto[2]/1e6,s[2]/1e6],color='red')
# plt.xlim([-8,8])
# plt.ylim([-8,8])
# ax.set_zlim(-8,8)
plt.xlabel(r"\textbf{$\sigma_1$* [MPa]}", fontsize=16)
plt.ylabel(r"\textbf{$\sigma_2$* [MPa]}", fontsize=16)
ax.set_zlabel(r"\textbf{$\tau_{12}$* [MPa]}", fontsize=16)
plt.title(r"$\textbf{Ellipsoid}$", fontsize=16)
plt.show()

