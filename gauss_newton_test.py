# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 00:43:23 2022

@author: hibad
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

def v2t(v):  
    t=np.asarray([[np.cos(v[2]), -np.sin(v[2]), v[0]],
                  [np.sin(v[2]), np.cos(v[2]), v[1]],
                  [0,0,1]])      
    return t

def t2v(t):  
    v=np.asarray([t[0,2], t[1,2], np.arctan2(t[1,0], t[0,0])])
    return v


def get_jacobian(x1, x2, z):
    z=np.linalg.inv(z)
    J1=[[z[0,1]*np.sin(x1[2])-z[0,0]*np.cos(x1[2]),-z[0,1]*np.cos(x1[2])-z[0,0]*np.sin(x1[2]),z[0,1]*(x1[0]*np.cos(x1[2])+x1[1]*np.sin(x1[2]))-z[0,0]*(x1[1]*np.cos(x1[2])-x1[0]*np.sin(x1[2]))-x2[0]*(z[0,1]*np.cos(x1[2])+z[0,0]*np.sin(x1[2]))+x2[1]*(z[0,0]*np.cos(x1[2]) - z[0,1]*np.sin(x1[2]))],
        [z[0,1]*np.cos(x1[2])+z[0,0]*np.sin(x1[2]),z[0,1]*np.sin(x1[2])-z[0,0]*np.cos(x1[2]),z[0,0]*(x1[0]*np.cos(x1[2])+x1[1]*np.sin(x1[2]))+z[0,1]*(x1[1]*np.cos(x1[2])-x1[0]*np.sin(x1[2]))-x2[0]*(z[0,0]*np.cos(x1[2])-z[0,1]*np.sin(x1[2]))-x2[1]*(z[0,1]*np.cos(x1[2]) + z[0,0]*np.sin(x1[2]))],
        [0,0,-1]]
 
 
    J2=[[z[0,0]*np.cos(x1[2])-z[0,1]*np.sin(x1[2]),z[0,1]*np.cos(x1[2])+z[0,0]*np.sin(x1[2]),0],
        [-z[0,1]*np.cos(x1[2])-z[0,0]*np.sin(x1[2]),z[0,0]*np.cos(x1[2])-z[0,1]*np.sin(x1[2]),0],
        [0,0,1]]
    return np.asarray(J1), np.asarray(J2)


#x=np.array([1, 3,0.0543, 4,1,0.26])
x=np.array([1, 3,0.0543, 5,5,1.2])
omega=np.eye(3)*999

for i in range(100):
    x1=np.copy(x[0:3])
    x2=np.copy(x[3:6])
    theta=0.26;
    zx=3;
    zy=-2;
    z=np.asarray([[np.cos(theta), -np.sin(theta), zx],
                  [np.sin(theta), np.cos(theta), zy],
                 [0,0,1]])
    J1, J2=get_jacobian(x1,x2,z)
    
    H=np.zeros((6,6))
    H[0:3,0:3]=J1.T@omega@J1
    H[3:6,3:6]=J2.T@omega@J2
    H[0:3,3:6]=J1.T@omega@J2
    H[3:6,0:3]=J2.T@omega@J1
    
    H[0:3, 0:3]+=np.eye(3)
    e=t2v(np.linalg.inv(z)@np.linalg.inv(v2t(x1))@v2t(x2))
    b=np.hstack((J1.T@omega@e, J2.T@omega@e))
    
    L=np.linalg.cholesky(H)
    y=solve_triangular(L,-b, lower=True)
    dx=solve_triangular(L.T, y)
   # dx=-np.linalg.inv(H)@b
    
    x+=dx
      

# print(x)
# H[0:3, 0:3]-=np.eye(3)
# print(H)