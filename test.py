# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:50:15 2022

@author: hibad
"""

import numpy as np
import matplotlib.pyplot as plt
from EKF_SLAM import EKF_SLAM
from scipy.io import savemat
from copy import deepcopy
from graph_SLAM import Graph_SLAM
from scipy.linalg import solve_triangular

R=np.asarray([[0.1,0],[0,0.1]])
Q=np.asarray([[0.01, 0],[0,0.01]])
mu_init=np.array([0.0,0.0,0.0])
slam=EKF_SLAM(mu_init, R, Q)
slam.estimate(1,[[5.03,0.03,0,1]])

#%%
mus=[deepcopy(slam.mu)]
sigmas=[deepcopy(slam.sigma)]
#u=[[7.24,-1.47], [7.65, -1.98], [6.36, -1.71], [7.34, -1.46]];

z=[[[5.07,-0.17,1,1]],  [[4.42,0.45,2,1]],[[4.39,1.01,3,1]],[[1.57,1.00,0,1]]]
u=[5*np.sqrt(2), -np.pi/2]
for i in range(4):
 #   mu, sigma, _=slam.update(1,z[i], u[i])
    mu, sigma, _=slam.update(1,z[i],u )
    mus.append(mu)
    sigmas.append(sigma)


dic={}
for i in range(5):
    dic[str(i)]={"mu" :mus[i], "sigma": sigmas[i]}
savemat("example.mat", {"mu" :mus, "sigma": sigmas})

#%%
def v2t(v):  
    t=np.asarray([[np.cos(v[2]), -np.sin(v[2]), v[0]],
                  [np.sin(v[2]), np.cos(v[2]), v[1]],
                  [0,0,1]])      
    return t

def t2v(t):  
    v=np.asarray([t[0,2], t[1,2], np.arctan2(t[1,0], t[0,0])])
    return v
class Front_end:
     class Node:
         def __init__(self, node_id, x, node_type):
             self.type=node_type
             self.x=deepcopy(x)
             self.id=node_id
             self.children={}
             self.parents={}
             self.local_map=[]
             
     class Edge:
         def __init__(self, node1, node2, Z, omega, edge_type):
             self.node1=node1
             self.node2=node2
             self.Z=Z
             self.omega=omega
             self.type=edge_type
             node1.children[node2.id]={"edge": self, "children": node2}
             node2.parents[node1.id]={"edge": self, "parents": node1}

         def get_error(self):
             e=t2v(np.linalg.inv(self.Z)@(np.linalg.inv(v2t(self.node1.x))@v2t(self.node2.x)))
             return e
             
     def __init__(self):
         self.nodes=[]
         self.edges=[]
         self.feature_nodes={}
     
     def add_node(self, x, node_type, feature_id=None, ):
         i=len(self.nodes)
         self.nodes.append(self.Node(i,x, node_type))
         if feature_id:
             self.feature_nodes[feature_id]=self.nodes[i]
         return i
     
     def add_edge(self, node_1_id, node_2_id, Z, omega, edge_type="odom"):
         self.edges.append(self.Edge(self.nodes[node_1_id], self.nodes[node_2_id],Z,omega,edge_type))
     
     
class Back_end:
     def get_jacobian(self, x1, x2, z):
         z=np.linalg.inv(z)
         J1=[[z[0,1]*np.sin(x1[2])-z[0,0]*np.cos(x1[2]),-z[0,1]*np.cos(x1[2])-z[0,0]*np.sin(x1[2]),z[0,1]*(x1[0]*np.cos(x1[2])+x1[1]*np.sin(x1[2]))-z[0,0]*(x1[1]*np.cos(x1[2])-x1[0]*np.sin(x1[2]))-x2[0]*(z[0,1]*np.cos(x1[2])+z[0,0]*np.sin(x1[2]))+x2[1]*(z[0,0]*np.cos(x1[2]) - z[0,1]*np.sin(x1[2]))],
             [z[0,1]*np.cos(x1[2])+z[0,0]*np.sin(x1[2]),z[0,1]*np.sin(x1[2])-z[0,0]*np.cos(x1[2]),z[0,0]*(x1[0]*np.cos(x1[2])+x1[1]*np.sin(x1[2]))+z[0,1]*(x1[1]*np.cos(x1[2])-x1[0]*np.sin(x1[2]))-x2[0]*(z[0,0]*np.cos(x1[2])-z[0,1]*np.sin(x1[2]))-x2[1]*(z[0,1]*np.cos(x1[2]) + z[0,0]*np.sin(x1[2]))],
             [0,0,-1]]
      
      
         J2=[[z[0,0]*np.cos(x1[2])-z[0,1]*np.sin(x1[2]),z[0,1]*np.cos(x1[2])+z[0,0]*np.sin(x1[2]),0],
             [-z[0,1]*np.cos(x1[2])-z[0,0]*np.sin(x1[2]),z[0,0]*np.cos(x1[2])-z[0,1]*np.sin(x1[2]),0],
             [0,0,1]]
         return np.asarray(J1), np.asarray(J2)
     def error_function(self, x1,x2,Z):
         return t2v(np.linalg.inv(Z)@(np.linalg.inv(v2t(x1))@v2t(x2)))

     def linearize(self,x, edges, idx_map):
         H=np.zeros((len(x), len(x)))
         b=np.zeros(len(x))
         for edge in edges:
             i=idx_map[str(edge.node1.id)]
             j=idx_map[str(edge.node2.id)]
             omega=edge.omega 
             Z=edge.Z
             A,B=self.get_jacobian(x[i:i+3], x[j:j+3], Z)
             H[i:i+3,i:i+3]+=A.T@omega@A
             H[j:j+3,j:j+3]+=B.T@omega@B
             H[i:i+3,j:j+3]+=A.T@omega@B
             H[j:j+3,i:i+3]+=H[i:i+3,j:j+3].T
             
             e=self.error_function(x[i:i+3], x[j:j+3], Z)
             b[i:i+3]+=A.T@omega@e
             b[j:j+3]+=B.T@omega@e
             
         return H,b
     
     def __init__(self):
         pass
     
     def node_to_vector(self, graph):
         idx_map={}
         x=np.zeros(3*len(graph.nodes))
         for i,node in enumerate(graph.nodes):
             x[3*i:3*i+3]=node.x
             idx_map[str(node.id)]=3*i
         return x, idx_map
     
     def linear_solve(self, A,b):
         L=np.linalg.cholesky(A)
         y=solve_triangular(L,b, lower=True)
         return solve_triangular(L.T, y)
     
     def update_nodes(self, graph,x, idx_map):
         for node in graph.nodes:
             idx=idx_map[str(node.id)]
             nodex=x[idx:idx+3]
             node.x=deepcopy(nodex)
         
     def optimize(self, graph):
         x, idx_map= self.node_to_vector(deepcopy(graph))
         H,b=self.linearize(x,deepcopy(graph.edges), idx_map)
         H[0:3,0:3]+=np.eye(3)

         dx=self.linear_solve(H,-b)
         x+=dx
         while np.max(dx)>0.001:
             H,b=self.linearize(x,deepcopy(graph.edges), idx_map)

             H[0:3,0:3]+=np.eye(3)
             dx=self.linear_solve(H,-b)
             x+=dx
         self.update_nodes(graph, x, idx_map)
         return x, H
     
graph=Front_end()
optimizer=Back_end()
x=[[5,-5,-np.pi/2],[0,-10,-np.pi], [-5,-5,np.pi/2], [0,0,0]]
node=graph.add_node(np.zeros(3), "pose")
feature_i=graph.add_node([5.02, 0.15 ,0], "feature")
f_omega=np.linalg.inv(np.array([[0.01, -0.007,0],[-0.007,0.25,0],[0,0,9999]]))
fu=np.asarray([[np.cos(1/2*u[1]), -1/2*u[0]*np.sin(1/2*u[1])],
               [np.sin(1/2*u[1]), 1/2*u[0]*np.cos(1/2*u[1])],
               [0, 1]])
p_omega=np.linalg.inv(fu@R@fu.T+np.eye(3))
graph.add_edge(node,feature_i, np.array([[1,0,5.02],[0,1,0.15],[0,0,1]]),f_omega , edge_type="measurement")
tf=v2t([5,-5,-np.pi/2])
for i in range(4):
    new_node=graph.add_node(x[i], "pose")
    graph.add_edge(node,new_node,tf ,p_omega , edge_type="measurement")
    node=new_node
    feature=[x[i][0]+z[i][0][0]*np.cos(x[i][2]+z[i][0][1]), x[i][1]+z[i][0][0]*np.sin(x[i][2]+z[i][0][1]), 0]
    j=graph.add_node(feature, "feature")
    tf=v2t([z[i][0][0]*np.cos(z[i][0][1]),z[i][0][1]*np.sin(z[i][0][1]),0])
    if not i==3:
        graph.add_edge(node,j,tf ,f_omega , edge_type="measurement")
    else:
        graph.add_edge(node,feature_i,tf ,f_omega , edge_type="measurement")

optimizer.optimize(graph)