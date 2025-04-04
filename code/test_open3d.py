# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:12:30 2024

@author: cbozonnet
"""


import open3d as o3d

import numpy as np
from numpy import pi, sin, cos, mgrid

########### compute ellipsoid radii ##############

epsY = 0.03 # threshold deformation
# elastic constants
Q11 = 177e6
Q22 = 12.8e6
Q12 = 4.9e6
# radii in the original frame
Y1 = Q11*epsY
Y2 = Q22*epsY
Y12 = Q12*epsY
# perform change of variable
alpha = 0.5*np.arctan(Y2**2/(Y2**2-Y1**2)) # angle of rotation
a_1 = np.sqrt(1/(cos(alpha)**2/Y1**2 + sin(alpha)**2/Y2**2 +cos(alpha)*sin(alpha)/Y1**2))
a_2 = np.sqrt(1/(sin(alpha)**2/Y1**2 + cos(alpha)**2/Y2**2 -cos(alpha)*sin(alpha)/Y1**2))
a_3 = Y12

###############" compute coordinates ################

dphi, dtheta = pi/30.0, pi/30.0 # angular step
[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta] # angular grid

xyz = np.zeros((np.size(phi), 3)) # data storage

center = [0, 0, 0] #ellipse center coordinates
radii = [a_1, a_2, a_3] # ellipse coefficient : [a_1,a_2,a_3]
# Compute variables using the parameteric equation
x = center[0] + radii[0] * np.cos(theta) * np.sin(phi)
y = center[1] + radii[1] * np.sin(theta) * np.sin(phi)
z = center[2] + radii[2] * np.cos(phi)
xyz[:, 0] = np.reshape(x, -1)
xyz[:, 1] = np.reshape(y, -1)
xyz[:, 2] = np.reshape(z, -1)


# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(xyz)

# Add color and estimate normals for better visualization.

pcd.paint_uniform_color([0.5, 0.5, 0.5])

pcd.estimate_normals()

pcd.orient_normals_consistent_tangent_plane(1)

print("Displaying Open3D pointcloud made using numpy array ...")

o3d.visualization.draw([pcd])


# Convert Open3D.o3d.geometry.PointCloud to numpy array.

xyz_converted = np.asarray(pcd.points)

print("Printing numpy array made using Open3D pointcloud ...")

print(xyz_converted)