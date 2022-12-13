import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
import keyboard
from Visualizer import Visualizer
from World import World
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from copy import deepcopy
import pickle
from ActorNet import ActorNet
from CriticNet import CriticNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
max_node=30
def v2t(v):  
    t=np.asarray([[np.cos(v[2]), -np.sin(v[2]), v[0]],
                  [np.sin(v[2]), np.cos(v[2]), v[1]],
                  [0,0,1]])      
    return t

def t2v(t):  
    v=np.asarray([t[0,2], t[1,2], np.arctan2(t[1,0], t[0,0])])
    return v

def angle_wrapping(theta):
    return deepcopy(np.arctan2(np.sin(theta), np.cos(theta)))    

class Q_Controller:
    def __init__(self, robot, world, critic_net, actor_net, gamma=0.99, manual=False):
        self.stm=[]
        self.ltm=[]
        self.robot=robot
        self.world=world
        
        self.critic_net=critic_net
        self.critic_optimizer=optim.Adam(critic_net.parameters())
        self.criterion=nn.MSELoss()
        
        self.actor_net=actor_net
        self.actor_optimizer=optim.Adam(actor_net.parameters())

        linear_velocity_space=[ 0.1 ,0.5, 1]
        angular_velocity_space=[-1,-0.5 ,0,0.5,1]
        self.action_space=np.array(np.meshgrid(linear_velocity_space, angular_velocity_space)).T.reshape(-1, 2)
        self.gamma=0.99

        self.manual=manual
        self.a_prev=[0,0]
        
    def _get_state(self):
        obstacle_dist ,idx=self.world.obstacles_tree.query(self.robot.odom[0:2], 1)
        obst=self.world.obstacles[idx].loc
        alpha=angle_wrapping(np.arctan2(obst[1]-self.robot.odom[1],obst[0]-self.robot.odom[0])-self.robot.odom[2])

        information=np.sum([1+np.tanh(np.log(np.linalg.det(node.H))) for node in robots[0].slam.front_end.pose_nodes])
        information+=np.sum([1+np.tanh(np.log(np.linalg.det(robots[0].slam.front_end.feature_nodes[key].H[0:2,0:2]))) for key in robots[0].slam.front_end.feature_nodes])
       
        return deepcopy({"pose": self.robot.odom,
                         "information": information,
                         "pose node count": len(robots[0].slam.front_end.pose_nodes), 
                         "feature node count": len(self.robot.slam.front_end.feature_nodes),
                         "proximity pose nodes": len(self.robot.slam.search_proximity_nodes(self.robot.odom, radius=1)),
                         "proximity feature nodes": len(self.robot.slam.search_proximity_features(self.robot.odom, radius=1)), 
                         "collided": self.world.collided,
                         "closest obstacle": (idx,obstacle_dist,alpha) ,
                         })
    
    def _get_state_feature(self, state):
        current_robot_pose=state["pose"]
        pose_node_count=state["pose node count"]
        feature_count=state["feature node count"]
        d_opt=state["information"]
        local_pose_node_count=state["proximity pose nodes"]
        local_feature_count=state["proximity feature nodes"]
        idx,obstacle_dist,alpha =state["closest obstacle"]
        feature=np.concatenate((current_robot_pose, [obstacle_dist, alpha, pose_node_count, feature_count, d_opt,local_pose_node_count,local_feature_count] ))
        return deepcopy(feature)
    
    def _reward(self, s, a):
        if s["collided"]:
            return -1000
        else:
            information_reward=s["information"]
            # feature_reward=s["feature node count"]
            if s["pose node count"]==max_node:
                return information_reward
            
            idx, dist, alpha=s["closest obstacle"]
         
            if dist<=0.5:
                obstacle_cost=1
            else:
                obstacle_cost=0
                
            rotation_cost=-0.05*(a[1])**2
            exploration_cost=2*s["proximity pose nodes"]
            if a[0]<0:
                backup_cost=-0.01*(a[0])**2
            else:
                backup_cost=0
                
            return 1+0.1*information_reward+rotation_cost+backup_cost-exploration_cost-obstacle_cost
        
    def Q_function(self, s,a):
        Q=torch.from_numpy(np.array([self._reward(s,a)]).astype(np.float32)).to(device)
        if not (s["collided"] or s["pose node count"]==max_node):
            feature=self._get_state_feature(s)
            vec=torch.from_numpy(feature.astype(np.float32)).to(device)
            Q+=self.gamma*self.critic_net.forward(vec)
        return Q
    

    def train_critic(self):
        critic_losses=[]
        for _ in range(10):
            batch=deepcopy(self.stm)
            if len(self.ltm)>1000:
                for _ in range(1000):
                    batch+=[deepcopy(self.ltm[np.random.randint(0,len(self.ltm))])]
                    
            target=[]
            inputs=[]
            print("collecting batch")
            for s,a,r,s_prime in batch:
                _, log_prob=self.policy(s_prime)
                prob=torch.exp(log_prob(torch.from_numpy(np.array(range(len(self.action_space)))).to(device)))
                Q=torch.cat([self.Q_function(s_prime, self.action_space[i]) for i in range(len(self.action_space))])
                Q_prime=(prob*Q).sum()
               # Q_prime=self.Q_function(s_prime, a_prime)
                target.append((self.gamma*Q_prime).unsqueeze(0))
                feature=self._get_state_feature(s)
                vec=torch.from_numpy(feature.astype(np.float32)).to(device)
                Q=self.critic_net(vec).float()
                inputs.append(self.gamma*Q)

            print("gradient descent")
            inputs=torch.cat(inputs, dim=0)      
            target=torch.cat(target, dim=0).detach()
            inputs=inputs.unsqueeze(0)
            target=target.unsqueeze(0)
            
            self.critic_optimizer.zero_grad()

            critic_loss=self.criterion(inputs,target)

            critic_loss.backward()

            self.critic_optimizer.step()

            critic_losses.append(critic_loss.item())
        return critic_losses
    
    def train_actor(self):
        actor_losses=[]
        batch=deepcopy(self.stm)
        if len(self.ltm)>1000:
            for _ in range(1000):
                batch+=[deepcopy(self.ltm[np.random.randint(0,len(self.ltm))])]
                
        advantage=[]
        log_probs=[]
        for s,a,r,s_prime in batch:
            _, log_prob=self.policy(s)
            
            prob=torch.exp(log_prob(torch.from_numpy(np.array(range(len(self.action_space)))).to(device)))
            Q=torch.cat([self.Q_function(s, self.action_space[i]) for i in range(len(self.action_space))])
            baseline=(prob*Q).sum()
            
            a_prime, _=self.policy(s_prime)
            advantage.append(r+self.gamma*self.Q_function(s_prime, a_prime)-baseline)
            a_idx=torch.from_numpy(np.where((self.action_space==a).all(1))[0]).to(device)
            log_probs.append(log_prob(a_idx).unsqueeze(0))

        advantage=torch.cat(advantage, dim=0).detach()

        log_probs = torch.cat(log_probs)
        
        self.actor_optimizer.zero_grad()

        actor_loss = -(log_probs * advantage.detach()).sum()

        actor_loss.backward()

        self.actor_optimizer.step()

        actor_losses.append(actor_loss.item())
        
        return actor_losses
    
    def train(self):
        cv2.destroyAllWindows()
        print("learning")
        critic_losses=self.train_critic()
        actor_losses=self.train_actor()

       
        return np.mean(critic_losses), np.mean(actor_losses)
        
        
    def collect(self):
        loss=(0,0)
        r=self._reward(self.state_prev, self.a_prev)
        state_prime=self._get_state()
        self.stm.append(deepcopy((self.state_prev, self.a_prev, r,state_prime )))
        if len(self.stm)>=2000:
                loss=self.train()
                self.ltm+=deepcopy(self.stm)
                self.ltm=self.ltm[-50000:]
                self.stm=[]
        return loss, r
    
    # def policy(self, state):
    #     action_space=deepcopy(self.action_space)
    #     Q=np.zeros(len(action_space))
    #     for i, a in enumerate(action_space):
    #         Q[i]=self.Q_function(state,a)
            
    #     if np.random.randint(10):
    #         a=action_space[np.argmax(Q)]
    #     else:
    #         ind = np.argpartition(Q, -5)[-5:]
    #         a=action_space[ind[np.random.randint(5)]]
    #     return deepcopy(a), np.max(Q)
    
    def policy(self,state):
        feature=self._get_state_feature(state)
        vec=torch.from_numpy(feature.astype(np.float32)).to(device)
        distribution = self.actor_net(vec)
        action = distribution.sample()
        action_idx=action.cpu().numpy()
        return self.action_space[action_idx], distribution.log_prob
    
    def teleop(self, a):
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
            return [0,0]
        a=deepcopy(a+du)
        a[0]=np.clip(a[0], -1,1)
        a[1]=np.clip(a[1], -1,1)
        print("trans ", a[0], "rot ", a[1])

        return a
    def collision_avoidance(self, a):
        action=deepcopy(a)
        d, idx =self.robot.world.obstacles_tree.query(self.robot.odom[0:2], 5)
        d=[(idx[i], dist) for i, dist in enumerate(d)  if dist <=0.5]
        alphas=[]
        w=0
        for i, dist in d:
            obst=deepcopy(self.robot.world.obstacles[i].loc)
            alpha=angle_wrapping(np.arctan2(obst[1]-self.robot.odom[1],obst[0]-self.robot.odom[0])-self.robot.odom[2])
            if (abs(alpha)<np.pi/2 and action[0]>0) or (abs(alpha)>np.pi/2 and action[0]<0):
                alphas.append(alpha/dist)
                w+=1/dist
                action[0]= action[0]*((1/10*(max(dist-0.3,0))))**(1/len(d)*abs(np.cos(alpha)))
        
        if len(alphas):
            alpha=np.mean(alphas)/w
            action[1]=-np.pi/2*np.sign(alpha)*np.cos(alpha)
                
        return action            
        
    def input_(self):
        state=self._get_state()
        self.state_prev=deepcopy(state)
        if self.manual:
            a=self.teleop(self.a_prev)
        else:
            a,_=self.policy(state)
        self.a_prev=deepcopy(a)
        a=self.collision_avoidance(a)
        self.robot.u=a

            
#%%  
critic_net=pickle.load( open( "Q_net.p", "rb" ) )
actor_net=pickle.load( open( "Actor_net.p", "rb" ) )

critic_losses=[]
actor_losses=[]
ltm=[]
#%%  
map_img = cv2.flip(cv2.imread("map_actual.png"),0)
map_resolution=0.12 #meter/pixel
origin=[0.6,0.6]

world_dt=0.01
process_interval=3
robots=[Robot([0.05,0.05], process_interval,slam_method="Graph_SLAM")]


world=World(map_img, origin, map_resolution, robots, random_feature=False)
controller=Q_Controller(robots[0], world,critic_net, actor_net)
robots[0].controller=controller
controller.ltm=ltm
vis=Visualizer(world,plot_resolution=100)

#%%
render=True
learned=False
for i in range(100000):
    episode_reward=[]
    if i<0:
        controller.manual=True
        render=True
    else:
        controller.manual=False

    print("Episode: ", i)
    loop=True
    world.reset()
    vis=Visualizer(world,plot_resolution=100)
    if learned:
        render=True
        learned=False
    while loop:  
        
        collided, processed=world.step(world_dt)
        if processed[0] or collided:
            loss, r=controller.collect()
            episode_reward.append(r)
            if loss[0]:
                critic_losses.append(loss[0])
                actor_losses.append(loss[1])
                
                print(loss)
                plt.figure()
                plt.plot(range(len(critic_losses)), critic_losses, "r.", alpha=0.1)
                plt.title("Critic Loss")
                plt.xlabel("Batch")
                plt.ylabel("MSE Loss")
                
                plt.figure()
                plt.plot(range(len(actor_losses)), actor_losses, "b.", alpha=0.1)
                plt.title("Actor Loss")
                plt.xlabel("Batch")
                plt.ylabel("Loss")
                plt.pause(0.05)
                learned=True
                ltm=controller.ltm
                pickle.dump( critic_net, open( "Q_net.p", "wb" ) )
                pickle.dump( actor_net, open(  "Actor_net.p", "wb" ) )

        loop=(not collided) and  (len(robots[0].slam.front_end.pose_nodes)<max_node) or (not processed[0])

        if render:
            vis.visualize() 
        if keyboard.is_pressed("esc"): 
             print('exit')
             loop=False 
        if render:
            cv2.waitKey(1)
            time.sleep(world_dt)
    render=False
    print("total reward: ",episode_reward)
    cv2.destroyAllWindows()
                

#%%
loop=True
robots=[Robot([0.05,0.1], process_interval,slam_method="Graph_SLAM")]
world=World(map_img, origin, map_resolution, robots, random_feature=False)
controller.robot=robots[0]
controller.world=world
robots[0].controller=controller
world.robots=robots

vis=Visualizer(world,plot_resolution=100, save_video=True)
    
while loop:  
    t=time.time()
    collided, processed=world.step(world_dt)
    loop=(not collided) and  (len(robots[0].slam.front_end.pose_nodes)<max_node)
    vis.visualize() 
    print(time.time()-t)
    if keyboard.is_pressed("esc"): 
         print('exit')
         loop=False 
         
    cv2.waitKey(1)
    time.sleep(world_dt)
    
vis.terminate()
cv2.destroyAllWindows()

