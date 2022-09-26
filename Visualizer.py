# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:11:01 2022

@author: hibad
"""
import cv2
from copy import deepcopy
import numpy as np

class Visualizer:
    def __init__(self,world, plot_resolution):
        self.world=world
        self.plot_resolution=plot_resolution
        bkg_map=deepcopy(world.bkg_map)
        w=int(bkg_map.shape[0]*self.plot_resolution*world.map_resolution)
        h=int(bkg_map.shape[1]*self.plot_resolution*world.map_resolution)
        bkg_map= cv2.resize(bkg_map, (w,h), interpolation=cv2.INTER_NEAREST)
        
        for obstacle in world.obstacles:
            if obstacle.is_feature:
                bkg_map=self.draw_features(bkg_map, obstacle.loc)
        self.bkg_map= bkg_map
    
  
    def state_to_pixel(self, state):
         return (int((state[0]+self.world.origin[0])*self.plot_resolution),int((state[1]+self.world.origin[1])*self.plot_resolution))
    
    def draw_features(self, image, loc):
         image=cv2.circle(image, self.state_to_pixel(loc) , int(0.05*self.plot_resolution),(33,67,101), -1)
         return image
         
     
    def draw_robot(self,image, world, pose, radius, color, alpha):
        start_point=self.state_to_pixel(pose)
        end_point=(int(start_point[0]+40*np.cos(pose[2])), int(start_point[1]+40*np.sin(pose[2])))
        image = cv2.arrowedLine(image, start_point, end_point,
                                         (0, 0, 0), 2)
        tem = deepcopy(image)
        tem= cv2.circle(tem, start_point, int(radius*self.plot_resolution),color, -1)
        image=cv2.addWeighted(image, 1-alpha, tem, alpha, 0)
        return image
    
    def draw_uncertainty(self, image, loc, covariance):
        loc=self.state_to_pixel(loc)
        u, s, vh =np.linalg.svd(covariance)
        angle=np.arctan2(u[0,1], u[0,0])
        axes=np.sqrt(s)*self.plot_resolution
        image = cv2.ellipse(img=image, center=loc ,axes=(int(axes[0]),int(axes[1])) ,
           angle=angle*180/np.pi, startAngle=0, endAngle=360, color=(255,255,0), thickness=2)
        return image 
    
    def draw_graph(self, image, graph):
        if len(graph.edges):
            thickness=[np.log(np.linalg.det(edge.omega)+1) for edge in graph.edges]
            thickness_range=np.max(thickness)-np.min(thickness)
            thickness=thickness/thickness_range*4
            thickness=thickness+1-np.min(thickness)
            thickness=np.clip(thickness, 1,5)
        for i,edge in enumerate(graph.edges):
            start_point=self.state_to_pixel(edge.node1.x[0:2])
            end_point=self.state_to_pixel(edge.node2.x[0:2])
            if edge.type=="odom":
                color=(255, 0, 0)
                image=cv2.line(image, start_point, end_point,color, int((thickness[i])))

            elif edge.type=="measurement":
                color=(0, 255, 255)    
            else:
                color=(0, 0, 255)    
                image=cv2.line(image, start_point, end_point,color, int((thickness[i])))

        for node in graph.nodes:
            x=self.state_to_pixel(node.x[0:2])
            if node.type=="pose":
                color=(0,0,255)
            else:
                color=(0,255,255)
            image=cv2.circle(image, x , int(0.05*self.plot_resolution),color, 2)
        return image
    def draw_map(self, image):
        for point in self.world.robots[0].map:
            loc=self.state_to_pixel(point)
            image=cv2.circle(image, loc , 1 ,(175,175,175), -11)
        return image
    
    def visualize(self):
        world=self.world
        image=deepcopy(self.bkg_map)
        if world.robots[0].slam.__class__.__name__=="Graph_SLAM":
            image=self.draw_graph(image, world.robots[0].slam.front_end)
    
        image=self.draw_robot(image,world, world.robots[0].x, world.robots[0].radius,  (131,46,75), 0.1)
        image=self.draw_robot(image,world, world.robots[0].odom, world.robots[0].radius, (131,46,75), 1)
        image=self.draw_uncertainty(image, world.robots[0].odom[0:2], world.robots[0].covariance[0:2, 0:2] )

        if world.robots[0].slam.__class__.__name__=="EKF_SLAM":
            image=self.draw_uncertainty(image, world.robots[0].odom[0:2], world.robots[0].covariance[0:2, 0:2] )
            for i in range(int((len(world.robots[0].odom)-3)/2)):
                loc=(world.robots[0].odom[2*i+3:2*i+5]+np.asarray(world.origin))*self.plot_resolution
                image= cv2.circle(image, (int(loc[0]),int(loc[1])), 2, (0,0,255), -1)
                image=self.draw_uncertainty(image, world.robots[0].odom[2*i+3:2*i+5], world.robots[0].covariance[2*i+3:2*i+5, 2*i+3:2*i+5])
        image=self.draw_map(image)
        cv2.imshow('test', cv2.flip(image, 0))


