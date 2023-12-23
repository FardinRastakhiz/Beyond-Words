# Omid Davar @ 2023


import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructorV2.GraphConstructor import GraphConstructor
from Scripts.DataManager.GraphConstructorV2.Anchors.Anchor import Anchor
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os



class VirtualConnector():

    def __init__(self , node_name = 'virtual' , num_of_nodes = 1 , target = 'token'):
        self.node_name = node_name
        self.settings = {'weight' : 0.5}
        self.num_of_nodes = num_of_nodes
        self.target = target
    def _build_initial_virtual_vector(self):
        return torch.zeros((self.num_of_nodes , self.nlp.vocab.vectors_length), dtype=torch.float32)
    def connect(self , graph : HeteroData):
        graph[self.node_name].x = self._build_initial_virtual_vector()
        for i in range(len(graph[self.target].x)):
            for j in range(self.num_of_nodes):
                target_virtual_edge_index.append([i , j])
                target_virtual_edge_attr.append(self.settings["weight"])
                virtual_target_edge_index.append([j , i])
                virtual_target_edge_attr.append(self.settings["weight"])
        graph[self.node_name , self.node_name + '_' + self.target , self.target].edge_index = torch.transpose(torch.tensor(virtual_target_edge_index, dtype=torch.int32) , 0 , 1) if len(virtual_target_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        graph[self.target , self.target + '_' + self.node_name , self.node_name].edge_index = torch.transpose(torch.tensor(target_virtual_edge_index, dtype=torch.int32) , 0 , 1) if len(target_virtual_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        graph[self.node_name , self.node_name + '_' + self.target , self.target].edge_attr = torch.tensor(virtual_target_edge_attr, dtype=torch.float32) 
        graph[self.target , self.target + '_' + self.node_name , self.node_name].edge_attr = torch.tensor(target_virtual_edge_attr, dtype=torch.float32)
        return graph
        
        
        
        
        

