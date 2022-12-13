# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:02:24 2022

@author: hibad
"""

import numpy as np
import keyboard
from copy import deepcopy
def angle_wrapping(theta):
    return deepcopy(np.arctan2(np.sin(theta), np.cos(theta)))   

class Tele_Controller:
    def input_(self):
        du=np.zeros(2)
        if keyboard.is_pressed("w"):
            du=np.asarray([0.05,0])
        if keyboard.is_pressed("x"):
            du=np.asarray([-0.05,0])
        if keyboard.is_pressed("a"):
            du=np.asarray([0,0.1])
        if keyboard.is_pressed("d"):
            du=np.asarray([0,-0.1])
        if keyboard.is_pressed("s"):
            self.robot.stop()
            return 
        
        self.u[0]+=du[0]
        self.u[1]+=du[1]
        self.u[0]=np.clip(self.u[0], -1,1)
        self.u[1]=np.clip(self.u[1], -1,1)
        print("trans ", self.u[0], "rot ", self.u[1])

        u=self.collision_avoidance(self.u) 
        
        self.robot.u=deepcopy(u)
    
    def collision_avoidance(self, a):
        action=deepcopy(a)
        d, idx =self.robot.world.obstacles_tree.query(self.robot.odom[0:2], 5)
        d=[(idx[i], dist) for i, dist in enumerate(d)  if dist <=0.5]
        for i, dist in d:
            obst=deepcopy(self.robot.world.obstacles[i].loc)
            alpha=angle_wrapping(np.arctan2(obst[1]-self.robot.odom[1],obst[0]-self.robot.odom[0])-self.robot.odom[2])
            if (abs(alpha)<np.pi/2 and action[0]>0) or (abs(alpha)>np.pi/2 and action[0]<0):
                action[0]= action[0]*((1/10*dist))**(1/len(d)*abs(np.cos(alpha)))
        return action                      
               
              
    def __init__(self, robot):
        self.robot=robot
        self.u=[0,0]