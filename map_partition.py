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
from region_graph import RegionGraph

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

#%% contiguous 
def expand(region, initial, action):
    next_states=[]
    for a in action:
        next_state=seed+a
        i=np.round(next_state.copy()).astype(int)
        if min(i)>=0 and (i<region.shape).all() and region[i[0], i[1], i[2]]==0.5:
            next_states.append(next_state)
    return next_states

def select_contiguous(region, seed, next_states, queue):
    for state in next_states:
        i=np.round(state.copy()).astype(int)
        region[i[0], i[1], i[2]]=1
        if seed in np.asarray(next_states):
            if not id(state) in map(id, queue):
                queue.append(state.copy())
        else:
            seed=state
    return region, seed, queue


action=[[1,0,0],[0,0,1],[-1,0,0],[0,0,-1], [0,1,0],[0,-1,0]]

for i in range(region.shape[0]):
    test=region[i,:,:,:]
    reachable_region=test*0.5
    seed=np.array(np.where(test==1)).T[0]
    queue=[]
    reachable_region[seed[0], seed[1], seed[2]]=1
    
    next_states=expand(reachable_region, seed, action)     
    reachable_region, seed, queue=select_contiguous(reachable_region, seed, next_states, queue)
    
    while len(queue)>0:
        seed_old=seed
        next_states=expand(reachable_region, seed, action)                 
        reachable_region, seed, queue=select_contiguous(reachable_region, seed, next_states, queue)
    
        if max(abs(seed_old-seed))<0.01:
            seed=queue[0]
            queue=queue[1:]
            
    # if np.where(reachable_region==0.5)[0]>10:
        
    for j in range(reachable_region.shape[2]):
        cv2.imshow("contiguous", cv2.resize(reachable_region[:,:,j], dsize=[500,500]))
        cv2.waitKey(100) 
    cv2.destroyAllWindows()

#%% reachbility 
test=region[0,:,:,:]
reachable_region=test*0.5
action=[[1,0,0],[0,0,1],[-1,0,0],[0,0,-1], [0,1,0],[0,-1,0]]
seed=np.array(np.where(test==1)).T[0]
queue=[]
reachable_region[seed[0], seed[1], seed[2]]=1
hist=[seed]
next_states=[]
for a in action:
    next_state=seed+a
    i=np.round(next_state.copy()).astype(int)
    if not sum([np.linalg.norm(next_state-x)<0.1 for x in hist]) and min(i)>=0 and(i<reachable_region.shape).all() and reachable_region[i[0], i[1], i[2]]:
        next_states.append(next_state)
        hist.append(next_state)
        
for state in next_states:
    i=np.round(state.copy()).astype(int)
    reachable_region[i[0], i[1], i[2]]=1
    if seed in np.asarray(next_states):
        if not sum([np.linalg.norm(state-x)<0.1 for x in queue]):
            queue.append(state.copy())
    else:
        seed=state
        #%%
j=0
while len(queue)>0:
    print(j)
    seed_old=seed
    next_states=[]
    for a in action:
        next_state=seed+a
        i=np.round(next_state.copy()).astype(int)
        if not sum([np.linalg.norm(next_state-x)<0.1 for x in hist]) and min(i)>=0 and(i<reachable_region.shape).all() and reachable_region[i[0], i[1], i[2]]:
            next_states.append(next_state)

            hist.append(next_state)
            
    for state in next_states:
        i=np.round(state.copy()).astype(int)
        reachable_region[i[0], i[1], i[2]]=1
        if seed in np.asarray(next_states):
            if not sum([np.linalg.norm(state-x)<0.1 for x in queue]):
                queue.append(state.copy())
        else:
            seed=state
                
    if max(abs(seed_old-seed))<0.01:
        seed=queue[0]
        queue=queue[1:]
    j+=1
for i in range(reachable_region.shape[2]):
    cv2.imshow("thing", cv2.resize(reachable_region[:,:,i], dsize=[500,500]))
    cv2.waitKey(100) 
cv2.destroyAllWindows()
#%% build graph
from region_graph import RegionGraph

    
    
name=["pillar", "I beam 1","I beam 2", "wall" ]

graph=RegionGraph([0.6,0.6], [0.12, 0.12, -2*np.pi/50])
graph.make_nodes(structures, region)
graph.make_edges()
graph.visualize()

pickle.dump( graph, open( "regionGraph.p", "wb" ) )
#%%
points=[]
colors=[]
for i,x in enumerate(np.linspace(-0.6, 5.4, 50)):
    for j, y in enumerate(np.linspace(-0.6, 5.4, 50)):
        for k,theta in enumerate(Theta):
            regions=graph.get_region(x, y, theta)
            if len(regions):
                points.append([x,y,theta])
                color=[np.array(cs.hsv_to_rgb(region/len(graph.nodes), 1, 1))for region in regions]
                colors.append((np.mean(color, axis=0)))
                    
                #%%
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors=o3d.utility.Vector3dVector(colors)
frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
o3d.visualization.draw_geometries([pcd, frame])
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
#                                                             voxel_size=0.12)
# o3d.visualization.draw_geometries([voxel_grid])
# tree=o3d.geometry.Octree()

# tree.convert_from_point_cloud(pcd)

# o3d.visualization.draw_geometries([tree])