# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:07:34 2022

@author: hibad
"""
import keyboard  
import numpy as np
from copy import deepcopy
from EKF_SLAM import EKF_SLAM
from graph_SLAM import Graph_SLAM
def angle_wrapping(theta):
    return deepcopy(np.arctan2(np.sin(theta), np.cos(theta)))    

def read_input(robot):
    if keyboard.is_pressed("w"):
        robot.tele_inputs(np.asarray([0.05,0]))
    if keyboard.is_pressed("x"):
        robot.tele_inputs(np.asarray([-0.05,0]))
    if keyboard.is_pressed("a"):
        robot.tele_inputs(np.asarray([0,0.1]))
    if keyboard.is_pressed("d"):
        robot.tele_inputs(np.asarray([0,-0.1]))
    if keyboard.is_pressed("s"):
        robot.stop()
        
class Camera:
    def __init__(self, robot, fov, depth,  range_noise, bearing_noise):
        self.fov=fov
        self.depth=depth         
        self.robot=robot 
        self.bearing_noise=bearing_noise
        self.range_noise=range_noise
        
    def measure(self):
        x=self.robot.x
        world=self.robot.world
        idx=self.robot.world.obstacles_tree.query_ball_point(x[0:2],self.depth)
        z=[[np.linalg.norm(world.obstacles[i].loc-x[0:2], 2), 
              angle_wrapping(np.arctan2(world.obstacles[i].loc[1]-x[1],world.obstacles[i].loc[0]-x[0])-x[2]),
              world.obstacles[i].id] for i in idx ]   
        
        z=[obs for obs in z if abs(obs[1])<=self.fov/2]  
        z=[[abs(np.random.normal(obs[0], self.range_noise)), angle_wrapping(np.random.normal(obs[1], self.bearing_noise)), obs[2] ]for obs in z]
        return z

    
class Robot:
    def __init__(self, input_noise,process_interval, slam_method="EKF_SLAM"):
        self.camera=Camera(self, 87*np.pi/180, 2, 0.1,0.1)
        self.x=np.zeros(6)
        self.x[2]=np.pi/2
        self.radius=0.3
        self.u=np.zeros(2)
        self.feature=[]
        self.input_noise=input_noise
        measurement_noise=[self.camera.range_noise, self.camera.bearing_noise]
        self.x_est=[[0,0]]
        self.odom=deepcopy(self.x[0:3])
        if slam_method=="EKF_SLAM":
            self.slam=EKF_SLAM(deepcopy(self.x[0:3]),np.diag(input_noise)**2, np.diag(measurement_noise)**2)
        else:
            self.slam=Graph_SLAM(deepcopy(self.x[0:3]),np.diag(input_noise)**2, np.diag(measurement_noise)**2,STM_length=3)
        self.covariance=np.zeros((3,3))
        self.camera_rate=0.03
        self.step_count=0
        self.process_interval=process_interval
        self.map=[]
        
    def tele_inputs(self, u):
        self.u[0]+=u[0]
        self.u[1]+=u[1]
        self.u[0]=np.clip(self.u[0], -1,1)
        self.u[1]=np.clip(self.u[1], -1,1)

    def stop(self):
        self.u=np.zeros(2)

    def input_eval(self,dt):
        trans=np.random.normal(self.u[0], self.input_noise[0])
        rot=np.random.normal(self.u[1], self.input_noise[1])
        if rot>=0.01:
            self.x[3]=1/dt*(-trans/rot*np.sin(self.x[2])+trans/rot*np.sin(self.x[2]+rot*dt))
            self.x[4]=1/dt*(trans/rot*np.cos(self.x[2])-trans/rot*np.cos(self.x[2]+rot*dt))
        else:
            self.x[3]=trans*np.cos(self.x[2]+1/2*self.world.dt*rot)
            self.x[4]=trans*np.sin(self.x[2]+1/2*self.world.dt*rot)
        
        self.x[5]=rot
        
    def observation(self, dt):
        obs=self.camera.measure()
        for i,z in enumerate(obs):
            z.append(self.world.obstacles[int(z[2])].is_feature)
        return obs
    
    def state_transition(self,dt):
        x=self.x
        dx=np.asarray([dt*x[3],
                       dt*x[4],
                       dt*x[5],0,0,0])
        self.x+=dx
        self.x[2]=angle_wrapping(self.x[2])
        

        self.x[0]=np.clip(x[0], self.world.bound[0,0],self.world.bound[0,1])
        self.x[1]=np.clip(x[1], self.world.bound[1,0],self.world.bound[1,1])
        
    def process(self, dt):
        if self.step_count >= self.process_interval:
            read_input(self)
            z=self.observation(self.process_interval*dt)
            self.odom, self.covariance, self.map=self.slam.update(self.process_interval*dt,z,deepcopy(self.u))
          
            self.step_count=0
        self.step_count+=1