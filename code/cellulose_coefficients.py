# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:23:42 2024

@author: cbozonnet
"""

import numpy as np


# Import the rigidity matrix

# cellulose I-beta Nishiyama 2003
# C = np.array([[20.9,9.2,7.3,0.3,0,0],
#             [9.3, 59.1, 12.1,-5.4,0,0],
#             [7.4,12.1,150.4,0.1,0,0],
#             [0.3,-5.1,0.2,3.8,0,0],
#             [0,0,0,0,2.9,25.8],
#             [0,0,0,0,25.8,10.5,]])

# Song 2021
C = np.array([[24.2,14.7,11.6,0,0,-1.5],
            [14.7, 100.6, 8.1,0,0,-10.6],
            [11.6,8.1,207.5,0,0,2.6],
            [0,0,0,16.2,-0.5,0],
            [0,0,0,-0.5,2.3,0],
            [-1.5,-10.6,2.6,0,0,5]])

# compute its determinent just in case
det = np.linalg.det(C)

if det != 0:
    # Inverse the matrix to get the compliance matrix
    S = np.linalg.inv(C)
    
    check = np.dot(S,C) # just to check that S.C=Identity matrix
    
    # # Create a new S matrix that is isotropic
    S_iso = np.zeros(S.shape) 
    
    # Fill-up its coefficients 
    # (watch out the subscripts, Python uses O-index)
    S_iso[0][0] = np.sqrt(S[0][0]*S[1][1]) # S'11²=S11*S22
    S_iso[1][1] = S_iso[0][0] # S'22=S'11
    S_iso[2][2] = S[2][2] # S'33=S33
    S_iso[0][2] = np.sqrt(S[0][2]*S[1][2]) # S'13²=S13*S23
    S_iso[1][2] = S_iso[0][2] # S'23=S'13
    S_iso[0][1] = S[0][1] # S'12=S12
    S_iso[3][3] = np.sqrt((S[3][3]*S[4][4])) # S'44²=S44*S55
    S_iso[4][4] = S_iso[3][3] #S'55=S'44
    S_iso[5][5] = S[5][5] # S'66=S66
    
    Ef = 1/S_iso[2][2] # Ef=E3=1/S33
    Gf = 1/S_iso[3][3] # Gf=G13=1/S44
    nuf = S_iso[0][2]/S_iso[2][2] #nuf = nu13 = S13/S33

        