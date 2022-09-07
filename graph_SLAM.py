# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 20:12:37 2022

@author: hibad
"""
import numpy as np
from copy import deepcopy
from EKF_SLAM import EKF_SLAM
from scipy.linalg import solve_triangular

def v2t(v):  
    t=np.asarray([[np.cos(v[2]), -np.sin(v[2]), v[0]],
                  [np.sin(v[2]), np.cos(v[2]), v[1]],
                  [0,0,1]])      
    return t

def t2v(t):  
    v=np.asarray([t[0,2], t[1,2], np.arctan2(t[1,0], t[0,0])])
    return v

class Graph_SLAM:
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
            
    def __init__(self,x_init, input_noise, measurement_noise, STM_length):
        self.front_end=self.Front_end()
        self.back_end=self.Back_end()
        self.x=deepcopy(x_init)
        self.sigma=np.zeros((3,3))
        self.current_node_id=self.front_end.add_node(self.x, "pose",[])
        self.R=input_noise
        self.Q=measurement_noise
        self.EKF_SLAM=EKF_SLAM(np.zeros(3),input_noise, measurement_noise)
        self.node_buffer_size=STM_length
        self.node_buffer=[0 for _ in range(STM_length)]
        self.loop_closure_thres=5
        self.local_map=[]
        self.global_map=[]
        
    def loop_closure(self, target, matching_features):
        print("loop close")
        current_node=self.front_end.nodes[self.current_node_id]
        z1=[]
        z2=[]
        for feature in matching_features:
            z1.append(t2v(current_node.children[feature]["edge"].Z)[0:2])
            z2.append(t2v(target.children[feature]["edge"].Z)[0:2])
        
        centroid1=np.mean(z1, axis=0)
        centroid2=np.mean(z2, axis=0)
        q1=z1-centroid1
        q2=z2-centroid2

        H=q1.T@q2
        U, _, V_t = np.linalg.svd(H)
        Rot = V_t.T@U.T
        dz = centroid2 - Rot@centroid1
        Z=np.hstack((Rot, dz.reshape((-1,1))))
        Z=np.vstack((Z,[0,0,1]))
        omega=np.eye(3)*0.001
        self.front_end.add_edge(deepcopy(self.current_node_id),target.id, Z, omega, edge_type="loop_closure")
        x, H=self.back_end.optimize(self.front_end)
        self.sigma=H[0:3,0:3]
        
    def loop_closure_detection(self):
        self_children=self.front_end.nodes[self.current_node_id].children
        feature_node_id=[self_children[node_key]["children"].id for node_key in self_children if self_children[node_key]["children"].type=="feature"]
        
        for node in self.front_end.nodes:
            if node.type=="pose":
                if node.id not in self.node_buffer and not node.id ==self.current_node_id :
                    features_id=[node.children[key]["children"].id for key in node.children if node.children[key]["children"].type=="feature"]
                    matches=[feature for feature in feature_node_id if feature in features_id]
                    if len(matches)>=self.loop_closure_thres:
                        self.loop_closure(node, matches)
                        
    def posterior_to_graph(self, mu, sigma,node_to_origin):
        features=deepcopy(self.EKF_SLAM.feature)
        for feature_id in features:
            idx=features[feature_id]
            x=np.hstack((mu[idx:idx+2],0))
            Z=v2t(x)
            x=t2v(node_to_origin@Z)
            if not feature_id in self.front_end.feature_nodes.keys():
                feature_node_id=self.front_end.add_node(x,"feature", feature_id)
            else:
                feature_node_id=self.front_end.feature_nodes[feature_id].id
                
            omega=np.eye(3)*0.01
            omega[0:2,0:2]=np.linalg.inv(sigma[idx:idx+2, idx:idx+2])
        
            self.front_end.add_edge(deepcopy(self.current_node_id),feature_node_id, Z,omega , edge_type="measurement")
        

    def global_map_assemble(self):
        self.global_map=[(v2t(node.x)@np.hstack((deepcopy(point),1)))[0:2]  for node in self.front_end.nodes for point in node.local_map ]
        if len(self.global_map)>1000:
            idx=np.random.choice(range(len(self.global_map)), 1000, replace=False)
            self.global_map=[self.global_map[i] for i in idx]
            
    def create_new_node(self, sigma, Z):
        self.front_end.nodes[self.current_node_id].local_map=deepcopy(self.local_map)
        self.local_map=[]
        new_node_id=self.front_end.add_node(self.x,"pose")
        omega=np.linalg.inv(sigma[0:3, 0:3]+np.eye(3)*0.001)
        self.front_end.add_edge(deepcopy(self.current_node_id),new_node_id, Z, omega)
        self.node_buffer.append(deepcopy(self.current_node_id))
        self.node_buffer=self.node_buffer[-self.node_buffer_size:]
        self.current_node_id=new_node_id      
        
    def update(self, dt, z, u): 
        x, sigma, _=self.EKF_SLAM.update(dt,z,u)
        node_x=deepcopy(self.front_end.nodes[self.current_node_id].x)
        node_to_origin=v2t(node_x)
        T=v2t(x[0:3])
        self.x=t2v(node_to_origin@T)
        self.local_map+=([[x[0]+obs[0]*np.cos(x[2]+obs[1]), x[1]+obs[0]*np.sin(x[2]+obs[1])] for obs in z])
        if np.linalg.norm(self.x[0:2]-node_x[0:2])>=1:
            self.posterior_to_graph(x, sigma, node_to_origin)
            self.loop_closure_detection()
            node_x=deepcopy(self.front_end.nodes[self.current_node_id].x)
            node_to_origin=v2t(node_x)
            self.x=t2v(node_to_origin@v2t(x[0:3]))
            self.create_new_node(sigma, T)
            self.global_map_assemble()
            self.EKF_SLAM=EKF_SLAM(np.zeros(3),deepcopy(self.R), deepcopy(self.Q))
        return deepcopy(self.x), deepcopy(self.sigma), deepcopy(self.global_map)