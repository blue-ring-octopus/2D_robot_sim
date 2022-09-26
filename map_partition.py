# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:27:04 2022

@author: hibad
"""
import cv2 
import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy
import open3d as o3d
import colorsys as cs
from numba import cuda 
import networkx as nx
import matplotlib.pyplot as plt
import pickle

def angle_wrapping(theta):
    return deepcopy(np.arctan2(np.sin(theta), np.cos(theta)))    
map_ref_img = cv2.flip(cv2.imread("map_prior.png"),0)
map_resolution=0.12 #meter/pixel
origin=[0.6,0.6]
all_rgb_codes = map_ref_img.reshape(-1, map_ref_img.shape[-1])
label_list=np.unique(all_rgb_codes, axis=0)
label_list=label_list[1:-1]
grayImage = cv2.cvtColor(map_ref_img, cv2.COLOR_BGR2GRAY)
_, blackAndWhiteImage = cv2.threshold(grayImage, 254, 255, cv2.THRESH_BINARY)
obs_loc=np.where(blackAndWhiteImage==0)
structures=[[] for _ in label_list]
for i in range(len(obs_loc[0])):
    x,y=(obs_loc[0][i],obs_loc[1][i])
    label=np.where((map_ref_img[x,y,:] == label_list).all(axis=1))[0]
    if len(label):
        structures[label[0]].append(np.asarray([obs_loc[0][i],obs_loc[1][i]]))

structure_trees=[KDTree(structure) for structure in structures if len(structure)]
lower_bound=[0,0]
upper_bound=np.array(map_ref_img.shape[0:2])
X=np.arange(lower_bound[0], upper_bound[0], 1)
Y=np.arange(lower_bound[1], upper_bound[1], 1)
Theta=np.linspace(-np.pi, np.pi, 50)
depth_max=2
depth_min=0.5
region=np.zeros((len(structures),len(X), len(Y), len(Theta)))
# for i in range(len(Theta)):
#     region[:,:,:,i]=blackAndWhiteImage
fov=87*np.pi/180

def ray_trace(alpha, step, target, point,k):
    r=np.linalg.norm(target-point)
    dp=np.asarray([step*np.cos(alpha), step*np.sin(alpha)])
    pt=point
    while np.linalg.norm(pt-point)<=r:
        if blackAndWhiteImage[int(pt[0]),int(pt[1])]==0 and not (map_ref_img[int(pt[0]),int(pt[1])]==label_list[k]).all():
            return False
        pt=dp+pt

    return True
        
def check_visible(loc, region, structure,i,j,k,l, theta):
    for point in structure:
        if np.linalg.norm(point-loc)>=depth_min/map_resolution:
            alpha=np.arctan2(point[1]-loc[1], point[0]-loc[0])
            if angle_wrapping(alpha-theta)<fov/2 and angle_wrapping(alpha-theta)>-fov/2:
                if ray_trace(alpha, 1, point, np.array([x,y]), k):
                    region[k,i,j,l]=1
                    return region   
    return region

# @cuda.jit()
# def raycast_kernel(d_color, d_dists, min_dist, max_dist):
#     i=cuda.grid(1)
#     nx=d_dists.shape[0]
#     if i<nx:
#         pass

# def raycast_par(X,Y,Theta, structure):
#     nx=X.shape[0]
#     ny=Y.shape[0]
#     nz=Theta.shape[0]
#     d_structure=cuda.to_device(structure)
#     d_region=cuda.device_array((nx,3),dtype=np.float64)
#     thread=(TPB)
#     blocks=(nx+TPB-1)//TPB
#     raycast_kernel[blocks, thread](d_color, d_dists, min_dist, max_dist)
    
#     return d_color.copy_to_host()              

for i,x in enumerate(X):
    for j, y in enumerate(Y):
        if blackAndWhiteImage[i,j]==255:
            for k,tree in enumerate(structure_trees):
                dist,_=structure_trees[k].query([x,y])
                if dist<=depth_max/map_resolution:
                    for l,theta in enumerate(Theta):
                        region=check_visible([x,y], region,structures[k] , i,j,k,l, theta)

#%% build graph
class RegionGraph:
    class Node:
        def __init__(self, node_id,name,grid):
            self.id=node_id
            self.name=name
            self.grid=grid
            self.edges=[]
    
    class Edge:
        def __init__(self, node1, node2, grid):
            self.grid=grid
            self.node1=node1
            self.node2=node2
            node1.edges.append(self)
            node2.edges.append(self)
    def __init__(self):
        self.nodes=[]
        self.edges=[]
        

    def make_nodes(self, structures, regions):
        for i,structure in enumerate(regions):
            self.nodes.append(self.Node(i,"test",structure))
        
    def make_edges(self):
        for i,node1 in enumerate(self.nodes[0:-1]):
            for node2 in self.nodes[i+1:]:
                overlap=node1.grid*node2.grid
                if np.sum(overlap):
                    self.Edge(node1,node2, overlap)
                    self.edges.append([node1.id, node2.id])
        
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.edges)
        nx.draw_networkx(G)
        plt.show()
        
    def get_region(self, x,y,theta):
        regions=[]
        for node in self.nodes:
            if node.grid[x,y,theta]:
                regions.append(node.id)
        return regions
    
    
name=["pillar", "I beam 1","I beam 2", "wall" ]

graph=RegionGraph()
graph.make_nodes(structures, region)
graph.make_edges()
graph.visualize()

pickle.dump( graph, open( "regionGraph.p", "wb" ) )
#%%
points=[]
colors=[]
for i,x in enumerate(X):
    for j, y in enumerate(Y):
        for k,theta in enumerate(Theta):
            regions=graph.get_region(i, j, k)
            if len(regions):
                points.append([i,j,k])
                color=[np.array(cs.hsv_to_rgb(region/len(graph.nodes), 1, 1))for region in regions]
                colors.append((np.mean(color, axis=0)))
                    
                #%%
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors=o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=1)
o3d.visualization.draw_geometries([voxel_grid])
# tree=o3d.geometry.Octree()

# tree.convert_from_point_cloud(pcd)

# o3d.visualization.draw_geometries([tree])