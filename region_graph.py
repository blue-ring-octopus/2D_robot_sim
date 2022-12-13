# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 01:03:36 2022

@author: hibad
"""
import numpy as np
import colorsys as cs
import networkx as nx
import matplotlib.pyplot as plt

class RegionGraph:
    class Node:
        def __init__(self, node_id,name,grid):
            self.id=node_id
            self.name=name
            self.grid=grid
            self.edges=[]
    
    class Edge:
        def __init__(self, node1, node2, grid):
            self.grid=grid
            self.node1=node1
            self.node2=node2
            node1.edges.append(self)
            node2.edges.append(self)
            
    def __init__(self, origin, scale):
        self.nodes=[]
        self.edges=[]
        self.origin=origin
        self.scale=scale

    def make_nodes(self, structures, regions):
        colorgrid=np.zeros(regions.shape+(3,))
        for i,region in enumerate(regions):
            self.nodes.append(self.Node(i,"test",region))
            colorgrid[i,:,:,:,:]=np.expand_dims(region, axis=3)*np.array(cs.hsv_to_rgb(i/len(regions),1,1))
        num=np.sum(regions, axis=0)
        self.colorgrid=np.nan_to_num(np.sum(colorgrid, axis=0)/np.expand_dims(num, axis=3))
        
    def make_edges(self):
        for i,node1 in enumerate(self.nodes[0:-1]):
            for node2 in self.nodes[i+1:]:
                overlap=node1.grid*node2.grid
                if np.sum(overlap):
                    self.Edge(node1,node2, overlap)
                    self.edges.append([node1.id, node2.id])
        
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.edges)
        nx.draw_networkx(G)
        plt.show()
        
    def get_region(self, x,y,theta):
        j=np.clip(int((x+self.origin[0])/self.scale[0]), 0, self.nodes[0].grid.shape[0]-1)
        i=np.clip(int((y+self.origin[1])/self.scale[1]), 0, self.nodes[0].grid.shape[1]-1)
        k=int((theta+np.pi/2)/self.scale[2])
        regions=[node.id for node in self.nodes if node.grid[i,j,k]]
        return regions