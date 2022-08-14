import keyboard  
import cv2
import time
import numpy as np
import copy
from scipy.spatial import KDTree
 
def angle_wrapping(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def visualize(world,plot_resolution):
    def draw_robot(image, world, pose, radius, color, alpha):
        start_point=(int((pose[0]+world.origin[0])*plot_resolution),int((pose[1]+world.origin[1])*plot_resolution))
        end_point=(int(start_point[0]+40*np.cos(pose[2])), int(start_point[1]+40*np.sin(pose[2])))
        image = cv2.arrowedLine(image, start_point, end_point,
                                         (0, 0, 0), 2)
        tem = copy.deepcopy(image)
        tem= cv2.circle(tem, start_point, int(radius*plot_resolution),color, -1)
        image=cv2.addWeighted(image, 1-alpha, tem, alpha, 0)
        return image
    
    def draw_uncertainty(image, loc, covariance):
        loc=(loc++world.origin[0])*plot_resolution
        u, s, vh =np.linalg.svd(covariance)
        angle=np.arctan2(u[0,1], u[0,0])
        axes=np.sqrt(s)*plot_resolution
        image = cv2.ellipse(img=image, center=(int(loc[0]), int(loc[1])) ,axes=(int(axes[0]),int(axes[1])) ,
           angle=angle*180/np.pi, startAngle=0, endAngle=360, color=(255,255,0), thickness=2)
        return image 
    
    image=copy.deepcopy(world.bkg_map)
    w=int(image.shape[0]*plot_resolution*world.map_resolution)
    h=int(image.shape[1]*plot_resolution*world.map_resolution)
    image= cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)

    image=draw_robot(image,world, world.robot.x, world.robot.radius,  (131,46,75), 0.1)
    
    image=draw_robot(image,world, world.robot.odom, world.robot.radius, (131,46,75), 1)
    
    image=draw_uncertainty(image, world.robot.odom[0:2], world.robot.covariance[0:2, 0:2] )
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
def rotational_matrix(theta):
    return np.asarray([[np.cos(theta), np.sin(theta)],
                       [-np.sin(theta), np.cos(theta)]])

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
            self.obstacles.append(Obstacle(obst_loc[-1], i))
        self.robot=Robot(self, [0.1,0.2])
        self.obstacles_tree=KDTree(obst_loc)
        
    def collision_check(self):
        dists, idx=self.obstacles_tree.query(self.robot.x[0:2],5)
        for i,dist in enumerate(dists):
            if dist<=(self.robot.radius+self.map_resolution*0.5):
                print("colision")
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
    def __init__(self, loc, obs_id):
        self.loc=loc
        self.id=obs_id

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
        z=[[np.linalg.norm(world.obstacles[i].loc-x[0:2]), 
              angle_wrapping(np.arctan2(world.obstacles[i].loc[1]-x[1],world.obstacles[i].loc[0]-x[0])-x[2]),
              i] for i in idx ]   
        
        z=[obs for obs in z if abs(obs[1])<=self.fov/2]  
        
        z=[[np.random.normal(obs[0], self.range_noise), np.random.normal(obs[1], self.bearing_noise), obs[2] ]for obs in z]
        return z
    
class EKF_SLAM:
    def __init__(self, robot):
        self.robot=robot
        self.mu=copy.deepcopy(robot.x[0:3])
        self.sigma=np.zeros(3)
        self.R=np.diag(robot.input_noise)**2
        self.Q=np.diag([robot.camera.range_noise**2, robot.camera.bearing_noise**2])
        self.feature=[]
        
    def predict(self, dt, u):
        self.mu+=np.asarray([dt*u[0]*np.cos(self.mu[2]),
                             dt*u[0]*np.sin(self.mu[2]),
                             dt*u[1]])
        self.mu[2]=angle_wrapping(self.mu[2])
        
        fx=np.asarray([[1,0,-dt*u[0]*np.sin(self.mu[2])], 
                       [0,1,dt*u[0]*np.cos(self.mu[2])],
                       [0,0,1]])
                       
        fu=np.asarray([[dt*np.cos(self.mu[2]), 0],
                       [dt*np.sin(self.mu[2]), 0],
                       [0, dt]])
        
        
        self.sigma=fx@self.sigma@fx.T +fu@self.R@fu.T
        print(self.sigma)
    def estimate(self,dt,z):
        pass
    
    def update(self,dt,z, u):
        self.predict(dt, u)
        self.estimate(dt,z)
        return self.mu, self.sigma
    
    
class Robot:
    def __init__(self, world, input_noise):
        self.camera=Camera(self, 87*np.pi/180, 2, 0.1,0.05)
        self.x=np.zeros(6)
        self.x[2]=np.pi/2
        self.radius=0.3
        self.world=world
        self.u=np.zeros(2)
        self.feature=[]
        self.input_noise=input_noise
        self.x_est=[[0,0]]
        self.odom=copy.deepcopy(self.x[0:3])
        self.slam=EKF_SLAM(self)
        self.covariance=np.zeros(6)
        
    def tele_inputs(self, u):
        self.u[0]+=u[0]
        self.u[1]+=u[1]
        self.u[0]=np.clip(self.u[0], -1,1)
        self.u[1]=np.clip(self.u[1], -1,1)

    def stop(self):
        self.u=np.zeros(2)

    def input_eval(self):
        trans=np.random.normal(self.u[0], self.input_noise[0])
        rot=np.random.normal(self.u[1], self.input_noise[1])
        
        self.x[3]=trans*np.cos(self.x[2])
        self.x[4]=trans*np.sin(self.x[2])
        self.x[5]=rot
        
    def observation(self, dt):
        obs=self.camera.measure()
        self.feature=[[self.x[0]+z[0]*np.cos(z[1]+self.x[2]),self.x[1]+z[0]*np.sin(z[1]+self.x[2])] for z in obs]
        
        self.odom, self.covariance=self.slam.update(dt,0,self.u)
        # self.odom+=np.asarray([dt*self.u[0]*np.cos(self.odom[2]),
        #                        dt*self.u[0]*np.sin(self.odom[2]),
        #                        dt*self.u[1]])
        
        # self.odom[2]=angle_wrapping(self.odom[2])
        
    def step(self, dt):
        self.observation(dt)
        x=self.x
        dx=np.asarray([dt*x[3],
            dt*x[4],
                      dt*x[5],0,0,0])
        self.x+=dx
        self.x[2]=angle_wrapping(self.x[2])
        

        self.x[0]=np.clip(x[0], self.world.bound[0,0],self.world.bound[0,1])
        self.x[1]=np.clip(x[1], self.world.bound[1,0],self.world.bound[1,1])

# class Graph_slam:
#     def __init__(self, x_init, max_iter):
#         self.x=[x_init]
#         self.max_iter=max_iter
    
#     def calc_edge(x1, y1, yaw1, x2, y2, yaw2, d1,
#                   angle1, d2, angle2, t1, t2):
#         edge = {}

#         tangle1 = angle_wrapping(yaw1 + angle1)
#         tangle2 = angle_wrapping(yaw2 + angle2)
#         tmp1 = d1 * np.cos(tangle1)
#         tmp2 = d2 * np.cos(tangle2)
#         tmp3 = d1 * np.sin(tangle1)
#         tmp4 = d2 * np.sin(tangle2)

#         edge.e[0, 0] = x2 - x1 - tmp1 + tmp2
#         edge.e[1, 0] = y2 - y1 - tmp3 + tmp4
#         edge.e[2, 0] = 0

#         Rt1 = rotational_matrix(tangle1)
#         Rt2 = rotational_matrix(tangle2)

#         sig1 = cal_observation_sigma()
#         sig2 = cal_observation_sigma()

#         edge.omega = np.linalg.inv(Rt1 @ sig1 @ Rt1.T + Rt2 @ sig2 @ Rt2.T)

#         edge.d1, edge.d2 = d1, d2
#         edge.yaw1, edge.yaw2 = yaw1, yaw2
#         edge.angle1, edge.angle2 = angle1, angle2
#         edge.id1, edge.id2 = t1, t2

#         return edge
    
#     def optimize(self, x_init, hz):
#         print("start graph based slam")

#         z_list = copy.deepcopy(hz)

#         x_opt = copy.deepcopy(x_init)
#         nt = x_opt.shape[1]
#         n = nt * x_init.shape[1]

#         for itr in range(self.max_iter):
#             edges = self.calc_edges(x_opt, z_list)

#             H = np.zeros((n, n))
#             b = np.zeros((n, 1))

#             for edge in edges:
#                 H, b = fill_H_and_b(H, b, edge)

#             # to fix origin
#             H[0:STATE_SIZE, 0:STATE_SIZE] += np.identity(STATE_SIZE)

#             dx = - np.linalg.inv(H) @ b

#             for i in range(nt):
#                 x_opt[0:3, i] += dx[i * 3:i * 3 + 3, 0]

#             diff = dx.T @ dx
#             print("iteration: %d, diff: %f" % (itr + 1, diff))
#             if diff < 1.0e-5:
#                 break

#         return x_opt
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
