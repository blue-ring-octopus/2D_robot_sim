import keyboard  
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.spatial import KDTree

def angle_wrapping(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def visualize(world,plot_resolution):

    image=copy.deepcopy(world.bkg_map)
    w=int(image.shape[0]*plot_resolution*world.map_resolution)
    h=int(image.shape[1]*plot_resolution*world.map_resolution)
    image= cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)

    start_point=(int((world.robot.x[0]+world.origin[0])*plot_resolution),int((world.robot.x[1]+world.origin[1])*plot_resolution))
    end_point=(int(start_point[0]+40*np.cos(world.robot.x[2])), int(start_point[1]+40*np.sin(world.robot.x[2])))
    image = cv2.arrowedLine(image, start_point, end_point,
                                     (0, 0, 0), 2)
    image= cv2.circle(image, start_point, int(world.robot.radius*plot_resolution), (131,46,75), -1)
    
    for obs in world.robot.feature:
        loc=(np.asarray(obs)+np.asarray(world.origin))*plot_resolution
        image= cv2.circle(image, (int(loc[0]),int(loc[1])), 2, (0,0,255), -1)

    
    cv2.imshow('test', cv2.flip(image, 0))

def read_input(robot):
    if keyboard.is_pressed("w"):
        robot.tele_inputs(np.asarray([0.1,0]))
    if keyboard.is_pressed("x"):
        robot.tele_inputs(np.asarray([-0.1,0]))
    if keyboard.is_pressed("a"):
        robot.tele_inputs(np.asarray([0,0.1]))
    if keyboard.is_pressed("d"):
        robot.tele_inputs(np.asarray([0,-0.1]))
    if keyboard.is_pressed("s"):
        robot.stop()

class World:
    def __init__(self, bkg_map, origin, map_resolution):
        self.bkg_map=bkg_map
        self.map_resolution=map_resolution        
        self.origin=origin
        self.bound=np.asarray([[-origin[0], map_resolution*bkg_map.shape[0]-origin[0]],
                               [-origin[1], map_resolution*bkg_map.shape[1]-origin[1]]])
        grayImage = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        _, blackAndWhiteImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        obs_loc=np.where(blackAndWhiteImage==0)
        self.obstacles=[]
        obst_loc=[]
        for i, idx in enumerate(obs_loc[0]):
            obst_loc.append(np.asarray(([(obs_loc[1][i]+0.5)*map_resolution-origin[0],(obs_loc[0][i]+0.5)*map_resolution-origin[1]])))
            self.obstacles.append(Obstacle(obst_loc[-1]))
        self.robot=Robot(self)
        self.obstacles_tree=KDTree(obst_loc)
        
    def collision_check(self):
        dists, idx=self.obstacles_tree.query(self.robot.x[0:2],5)
        for i,dist in enumerate(dists):
            if dist<=(self.robot.radius+self.map_resolution*0.5):
                if not dist==0:
                    collision=1/dist*np.asarray([self.obstacles[idx[i]].loc[0]-self.robot.x[0], self.obstacles[idx[i]].loc[1]-self.robot.x[1]])
                    if np.dot(self.robot.x[3:5], collision)>=0:                        
                        self.robot.x[3:5]=self.robot.x[3:5]-np.dot(self.robot.x[3:5], collision)*collision

       # print(dist)
    def step(self, dt):
        self.robot.input_eval()
        self.collision_check()
        self.robot.step(dt)
        
class Obstacle:
    def __init__(self, loc):
        self.loc=loc

class Camera:
    def __init__(self, robot, fov, depth, bearing_noise, range_noise):
        self.fov=fov
        self.depth=depth         
        self.robot=robot 
        self.bearing_noise=bearing_noise
        self.range_noise=range_noise
        
    def observe(self):
        x=self.robot.x
        world=self.robot.world
        idx=self.robot.world.obstacles_tree.query_ball_point(x[0:2],self.depth)
        z=[[np.linalg.norm(world.obstacles[i].loc-x[0:2]), 
              angle_wrapping(np.arctan2(world.obstacles[i].loc[1]-x[1],world.obstacles[i].loc[0]-x[0])-x[2])] for i in idx ]   
        
        z=[obs for obs in z if abs(obs[1])<=self.fov/2]  
        
        z=[[np.random.normal(obs[0], self.range_noise), np.random.normal(obs[1], self.bearing_noise) ]for obs in z]
        return z
    
class Robot:
    def __init__(self, world):
        self.camera=Camera(self, 87*np.pi/180, 1.5, 0.1,0.1)
        self.x=np.zeros(6)
        self.x[2]=np.pi/2
        self.radius=0.3
        self.world=world
        self.u=np.zeros(2)
        self.feature=[]
        
    def tele_inputs(self, u):
        self.u[0]+=u[0]
        self.u[1]+=u[1]
        self.u[0]=np.clip(self.u[0], -1,1)
        self.u[1]=np.clip(self.u[1], -1,1)

    
    def stop(self):
        self.u=np.zeros(2)

    def input_eval(self):
        self.x[3]=self.u[0]*np.cos(self.x[2])
        self.x[4]=self.u[0]*np.sin(self.x[2])
        self.x[5]=self.u[1]  
        
    def observation(self):
        obs=self.camera.observe()
        self.feature=[[self.x[0]+z[0]*np.cos(z[1]+self.x[2]),self.x[1]+z[0]*np.sin(z[1]+self.x[2])] for z in obs]
        
        
    def step(self, dt):
        self.observation()
        x=self.x
        dx=np.asarray([dt*x[3],
            dt*x[4],
                      dt*x[5],0,0,0])
        self.x+=dx
        self.x[2]=angle_wrapping(self.x[2])
        

        self.x[0]=np.clip(x[0], self.world.bound[0,0],self.world.bound[0,1])
        self.x[1]=np.clip(x[1], self.world.bound[1,0],self.world.bound[1,1])
        
map_img = cv2.flip(cv2.imread("map.png"),0)
map_resolution=0.12 #meter/pixel
origin=[0.6,0.6]

world=World(map_img, origin, map_resolution)
dt=0.03
tree=world.obstacles_tree
#%%
loop=True

while loop:  # making a loop
  #  try:
        read_input(world.robot)
        world.step(dt)
        print("trans ", world.robot.u[0], "rot ", world.robot.u[1])
        visualize(world,plot_resolution=100) #pixel per meter
        if keyboard.is_pressed("esc"):  # if key 'q' is pressed 
             print('exit')
             loop=False 
        cv2.waitKey(int(dt*1000))
  #  except Exception as e:
  #      print(e)
   #     cv2.destroyAllWindows()
   #     break
cv2.destroyAllWindows()
