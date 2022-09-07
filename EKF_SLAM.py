# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 22:59:07 2022

@author: hibad
"""
import copy
import numpy as np

def angle_wrapping(theta):
    return copy.deepcopy(np.arctan2(np.sin(theta), np.cos(theta)))    
class EKF_SLAM:
    def __init__(self, mu_init, input_noise, measurement_noise):
        self.mu=copy.deepcopy(mu_init)
        self.sigma=np.zeros((3,3))
        self.R=input_noise
        self.Q=measurement_noise
        self.feature={}

    def predict(self, dt, u):
        F=np.zeros((3,self.mu.shape[0]))
        F[0:3,0:3]=np.eye(3)
        
        if u[1]>=0.01:
            self.mu[0:3]+=np.asarray([-u[0]/u[1]*np.sin(self.mu[2])+u[0]/u[1]*np.sin(self.mu[2]+u[1]*dt),
                                     u[0]/u[1]*np.cos(self.mu[2])-u[0]/u[1]*np.cos(self.mu[2]+u[1]*dt),
                                     dt*u[1]])
            self.mu[2]=angle_wrapping(self.mu[2])
            
            fx=np.eye(self.mu.shape[0])+F.T@np.asarray([[0,0,-u[0]/u[1]*np.cos(self.mu[2])+u[0]/u[1]*np.cos(self.mu[2]+u[1]*dt)], 
                                                        [0,0,-u[0]/u[1]*np.sin(self.mu[2])+u[0]/u[1]*np.sin(self.mu[2]+u[1]*dt)],
                                                        [0,0,0]])@F
            
            fu=np.asarray([[-1/u[1]*np.sin(self.mu[2])+1/u[1]*np.sin(self.mu[2]+u[1]*dt), 
                           u[0]/(u[1]**2)*(np.sin(self.mu[2])+dt*u[1]*np.cos(self.mu[2]+u[1]*dt)-np.sin(self.mu[2]+u[1]*dt))],
                           [1/u[1]*np.cos(self.mu[2])-1/u[1]*np.cos(self.mu[2]+u[1]*dt), 
                        u[0]/(u[1]**2)*(-np.cos(self.mu[2])+dt*u[1]*np.sin(self.mu[2]+u[1]*dt)+np.cos(self.mu[2]+u[1]*dt))],
                           [0, dt]])
        else:
            self.mu[0:3]+=np.asarray([dt*u[0]*np.cos(self.mu[2]),
                                 dt*u[0]*np.sin(self.mu[2]),
                                 dt*u[1]])
            self.mu[2]=angle_wrapping(self.mu[2])
            
            fx=np.eye(self.mu.shape[0])+F.T@np.asarray([[0,0,-dt*u[0]*np.sin(self.mu[2])], 
                                                        [0,0,dt*u[0]*np.cos(self.mu[2])],
                                                        [0,0,0]])@F
                           
            fu=np.asarray([[dt*np.cos(self.mu[2]), 0],
                           [dt*np.sin(self.mu[2]), 0],
                           [0, dt]])
         
        self.sigma=(fx)@self.sigma@(fx.T)+F.T@(fu)@self.R@(fu.T)@F

    def estimate(self,dt,feature):
        if len(feature)>0:
            for z in feature:
                if z[3]:
                    if not str(z[2]) in self.feature.keys():
                        loc=np.asarray([self.mu[0]+z[0]*np.cos(z[1]+self.mu[2]),
                                       self.mu[1]+z[0]*np.sin(z[1]+self.mu[2])])
                        self.feature[str(z[2])]=self.mu.shape[0]
                        self.mu=np.hstack((copy.deepcopy(self.mu), loc))
                        sigma_new=np.diag(np.ones(self.sigma.shape[0]+2)*9999)
                        sigma_new[0:self.sigma.shape[0], 0:self.sigma.shape[0]]=copy.deepcopy(self.sigma)
                        self.sigma=sigma_new
            KH=np.eye(self.mu.shape[0])    
            mu=copy.deepcopy(self.mu)
            test=0
            for z in feature:    
                if z[3]:
                    idx=self.feature[str(z[2])]
                    loc=mu[idx:idx+2]
                    dx=loc-mu[0:2]
                    q=(dx@dx)
                    r=np.sqrt(q)
                    z_bar=np.asarray([r,
                                      angle_wrapping(np.arctan2(dx[1],dx[0])-mu[2])])
                    F=np.zeros((5,mu.shape[0]))
                    F[0:3,0:3]=np.eye(3)
                    F[3, idx]=1
                    F[4, idx+1]=1

                    H=(np.asarray([[-dx[0]/r, -dx[1]/r, 0,dx[0]/r, dx[1]/r],
                                  [dx[1]/q, -dx[0]/q, -1, -dx[1]/q, dx[0]/q]]))@F
                    
                    K=self.sigma@(H.T)@np.linalg.inv((H@self.sigma@(H.T)+self.Q))
                    KH+=-K@H
                    
                    dz=z[0:2]-z_bar
                    dz[1]=angle_wrapping(dz[1])
                    self.mu+=K@(dz)
                    test+=K@(dz)
                    self.mu[2]=angle_wrapping(self.mu[2])
            
            self.sigma=(KH)@(self.sigma)
        
    def update(self,dt,z, u):
        self.predict(dt, u)
        self.estimate(dt,z)
        return self.mu, self.sigma, []
    