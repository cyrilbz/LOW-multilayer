# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:43:01 2024

@author: cbozonnet
"""

import numpy as np
from timeit import Timer

from functions import *

num_layers = [5]

loop_times = []
vec_times = []

for n in num_layers:
  # Create sample angles array
  pi=np.pi
  angles = np.random.rand(n)
  angles = np.array((pi/2,pi/2))

  # Time loop-based implementation
  loop_timer = Timer(lambda: rotation_matrix_loop(angles.copy()))
  loop_time = loop_timer.timeit(number=100)
  loop_times.append(loop_time)

  # Time vectorized implementation
  vec_timer = Timer(lambda: rotation_matrix_vectorized(angles.copy()))
  vec_time = vec_timer.timeit(number=100)
  vec_times.append(vec_time)

print("Number of Layers  | Loop Time (s) | Vectorized Time (s)")
print("------------------|----------------|--------------------")
for i, n in enumerate(num_layers):
  print(f"{n}               | {loop_times[i]:.6f} | {vec_times[i]:.6f}")
  
out =  rotation_matrix_vectorized(angles.copy()) # a nlx3x3 rotation matrix
 
eps = np.array((1,0,0))

epsMF=out@eps # nl x 3 matrix
tr = np.transpose(epsMF) # 3 x nl matrix

p=parameters()

sig = p.C_el @ tr # a 3 x nl matrix

contra =  rotation_matrix_vectorized(-angles.copy()) # a nlx3x3 rotation matrix

# sig_Ref = contra @ sig.T
sig_Ref = np.zeros((3, len(angles)))
for i in range(len(angles)):
  # Perform multiplication for each layer
  sig_Ref[:, i] = contra[i, :, :] @ sig[:, i]
  
sig_Ref_vec = contra @ sig.T
  
  