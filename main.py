import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
import keyboard
from colorsys import hsv_to_rgb
from Visualizer import Visualizer
from World import World
from scipy.spatial import KDTree
from keyboard_controller import Tele_Controller
          
    
map_img = cv2.flip(cv2.imread("map_actual.png"),0)
map_resolution=0.12 #meter/pixel
origin=[0.6,0.6]

world_dt=0.01
process_interval=3
robots=[Robot([0.05,0.1], process_interval,slam_method="Graph_SLAM")]
tele_controller=Tele_Controller(robots[0])
robots[0].controller=tele_controller

#robots=[Robot([0.05,0.1], process_interval ,slam_method="EKF_SLAM")]
world=World(map_img, origin, map_resolution, robots, random_feature=False)
vis=Visualizer(world,plot_resolution=100)
#%%
loop=True
render=True
max_node=100
while loop:  
  #  try:
        collided, _ =world.step(world_dt)
        loop=(not collided) and (len(robots[0].slam.front_end.pose_nodes)<max_node)
        if render:
            vis.visualize() 
        if keyboard.is_pressed("r"): 
            world.reset()
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