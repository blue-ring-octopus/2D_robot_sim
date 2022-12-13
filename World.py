# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:13:50 2022

@author: hibad
"""
import cv2
import numpy as np
from scipy.spatial import KDTree


class World:
    def __init__(self, bkg_map, origin, map_resolution, agents, random_feature=True):

        self.bkg_map=bkg_map
        self.map_resolution=map_resolution        
        self.origin=origin
        self.bound=np.asarray([[-origin[0], map_resolution*bkg_map.shape[0]-origin[0]],
                               [-origin[1], map_resolution*bkg_map.shape[1]-origin[1]]])
        grayImage = cv2.cvtColor(bkg_map, cv2.COLOR_BGR2GRAY)
        _, blackAndWhiteImage = cv2.threshold(grayImage, 250, 255, cv2.THRESH_BINARY)
        obs_loc_pixel=np.where(blackAndWhiteImage==0)
        self.obstacles=[]
        obst_loc=[]
        for i  in range(len(obs_loc_pixel[0])):
            obst_loc.append(np.asarray(([(obs_loc_pixel[1][i]+0.5)*map_resolution-origin[0],(obs_loc_pixel[0][i]+0.5)*map_resolution-origin[1]])))
            if random_feature:
                self.obstacles.append(Obstacle(obst_loc[-1], obs_id=i, is_feature=(np.random.uniform(0,1)<0.2)))
            else:
                is_feature= not (bkg_map[obs_loc_pixel[0][i], obs_loc_pixel[1][i]]==[ 0,  0 ,0]).all()
                self.obstacles.append(Obstacle(obst_loc[-1], obs_id=i, is_feature=is_feature))    
        self.robots=agents
        for robot in self.robots:
            robot.world=self
        self.obstacles_tree=KDTree(obst_loc)
        self.reset()
        
    def reset(self):
        self.collided=False
        self.t=0
        for robot in self.robots:
            robot.reset()
        self._collision_check()

    
    def _collision_check(self):
        for robot in self.robots:
            dists, idx=self.obstacles_tree.query(robot.x[0:2],5)
            self.closest_obstacles=self.obstacles[idx[np.argmin(dists)]].loc
            for i,dist in enumerate(dists):
                if dist<=(robot.radius+self.map_resolution*0.5):
                    self.collided=True
                    if not dist==0:
                        collision=1/dist*np.asarray([self.obstacles[idx[i]].loc[0]-robot.x[0], self.obstacles[idx[i]].loc[1]-robot.x[1]])
                        if np.dot(robot.x[3:5], collision)>=0:                        
                            robot.x[3:5]=robot.x[3:5]-np.dot(robot.x[3:5], collision)*collision
    
    
    
    def step(self, dt):
        processed=[]
        self.dt=dt
        for robot in self.robots:
            robot.input_eval(dt)
        self._collision_check()
        for robot in self.robots:
            robot.state_transition(dt)
            processed.append(robot.process(dt))
        self.t+=dt
        return self.collided, processed

class Obstacle:
    def __init__(self, loc, obs_id,is_feature):
        self.loc=loc
        self.id=obs_id
        self.is_feature=is_feature
