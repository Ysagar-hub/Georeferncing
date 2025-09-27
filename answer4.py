# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 20:52:56 2025

@author: IITG sagar
"""


import numpy as np

# Satellite positions and pseudoranges
x1, y1, z1, p1 = 15470964, -1180726.85, 21541839, 20260438.9
x2, y2, z2, p2 = 19603002, 4726671.62, 17059949, 20264387.99
x3, y3, z3, p3 = 3017917, 15760014.6, 21367886, 23104936.52
x4, y4, z4, p4 = 180842.8, -15551720.3, 21714117, 23382913.05
x5, y5, z5, p5 = 25616942, 7756572.54, 336879.3, 23101783.81

c = 299792458.0  # speed of light

def funA(x, y, z):
    # Jacobian matrix 5x4
    return np.array([
        [-(x-x1)/np.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2),
         -(y-y1)/np.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2),
         -(z-z1)/np.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2),
         -c],
        [-(x-x2)/np.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2),
         -(y-y2)/np.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2),
         -(z-z2)/np.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2),
         -c],
        [-(x-x3)/np.sqrt((x-x3)**2 + (y-y3)**2 + (z-z3)**2),
         -(y-y3)/np.sqrt((x-x3)**2 + (y-y3)**2 + (z-z3)**2),
         -(z-z3)/np.sqrt((x-x3)**2 + (y-y3)**2 + (z-z3)**2),
         -c],
        [-(x-x4)/np.sqrt((x-x4)**2 + (y-y4)**2 + (z-z4)**2),
         -(y-y4)/np.sqrt((x-x4)**2 + (y-y4)**2 + (z-z4)**2),
         -(z-z4)/np.sqrt((x-x4)**2 + (y-y4)**2 + (z-z4)**2),
         -c],
        [-(x-x5)/np.sqrt((x-x5)**2 + (y-y5)**2 + (z-z5)**2),
         -(y-y5)/np.sqrt((x-x5)**2 + (y-y5)**2 + (z-z5)**2),
         -(z-z5)/np.sqrt((x-x5)**2 + (y-y5)**2 + (z-z5)**2),
         -c]
    ], dtype=float)

def funB(x, y, z, t):
    # Residual vector 5x1
    return np.array([
        -p1 + np.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2) + c*t,
        -p2 + np.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2) + c*t,
        -p3 + np.sqrt((x-x3)**2 + (y-y3)**2 + (z-z3)**2) + c*t,
        -p4 + np.sqrt((x-x4)**2 + (y-y4)**2 + (z-z4)**2) + c*t,
        -p5 + np.sqrt((x-x5)**2 + (y-y5)**2 + (z-z5)**2) + c*t
    ], dtype=float).reshape(-1,1)  # 5x1 column vector

# Initial guess
x0 = np.array([[1.0], [1.0], [1.0], [1.0]])
estimates = [x0.flatten()]

# Iterative solution
for i in range(5):
    X, Y, Z, t = x0.flatten()  # unpack for clarity
    amat = funA(X, Y, Z)
    bmat = funB(X, Y, Z, t)
    
    At_A = amat.T @ amat
    if np.isclose(np.linalg.det(At_A), 0):
       print("Warning: A.T @ A is singular (non-invertible). "
            "The Normal Equation cannot be solved directly.")
       print("Consider using alternative methods like `np.linalg.lstsq()` or regularization.")
       delX = np.nap
    else:
        # Calculate the inverse and solve for X1 iteration 1
        delX =  np.linalg.inv(amat.T @ amat) @ amat.T @ bmat 

    
    x0 = x0 + delX

    err = np.linalg.norm(delX)/np.linalg.norm(x0)*100
    print(f"Iteration {i+1}, Error % = {err:.6f}")
    print(f"Estimate: X={x0[0,0]:.3f}, Y={x0[1,0]:.3f}, Z={x0[2,0]:.3f}, t={x0[3,0]:.6f}")

    estimates.append(x0.flatten())

# Convert estimates to array
estimates = np.array(estimates)
#print("\nAll Estimates:\n", estimates)
