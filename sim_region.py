# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 00:59:31 2022

@author: hibad
"""
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
import keyboard
from colorsys import hsv_to_rgb
from Visualizer_region import Visualizer
from World import World
from scipy.spatial import KDTree
from region_graph import RegionGraph
import pickle
#%%
regionGraph=pickle.load( open( "regionGraph.p", "rb" ) )
map_img = cv2.flip(cv2.imread("map_actual.png"),0)
map_resolution=0.12 #meter/pixel
origin=[0.6,0.6]

world_dt=0.01
process_interval=3

robots=[Robot([0.05,0.1], process_interval ,slam_method="Graph_SLAM")]
world=World(map_img, origin, map_resolution, robots)
vis=Visualizer(world,plot_resolution=100, region=regionGraph)
#%%
loop=True
while loop:  
  #  try:
        world.step(world_dt)
        print("trans ", world.robots[0].u[0], "rot ", world.robots[0].u[1])
        vis.visualize() 
        if keyboard.is_pressed("esc"): 
             print('exit')
             loop=False 
        cv2.waitKey(1)
        time.sleep(world_dt)
        

  #  except Exception as e:
  #      print(e)
   #     cv2.destroyAllWindows()
   #     break
cv2.destroyAllWindows()
pointcloud=robots[0].map
#%%
map_ref_img = cv2.flip(cv2.imread("map_prior.png"),0)
grayImage = cv2.cvtColor(map_ref_img, cv2.COLOR_BGR2GRAY)
_, blackAndWhiteImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
obs_loc=np.where(blackAndWhiteImage==0)
obstacles=[]
obst_loc=[]
for i, idx in enumerate(obs_loc[0]):
    obst_loc.append(np.asarray(([(obs_loc[1][i]+0.5)*map_resolution-origin[0],(obs_loc[0][i]+0.5)*map_resolution-origin[1]])))
    
tree=KDTree(np.asarray(obst_loc))

dists, idxs=tree.query(pointcloud)
q3=np.quantile(dists, 0.97)
dists[dists>q3]=q3
hue=(dists-min(dists))/(max(dists)-min(dists))
for i,point in enumerate(pointcloud):
    plt.plot(point[0], point[1], '.', color=hsv_to_rgb(0.66-0.66*hue[i], 1,1))