# Omid Davar @ 2023


import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructorV2.GraphConstructor import GraphConstructor
from Scripts.DataManager.GraphConstructorV2.Anchors.Anchor import Anchor
from Scripts.DataManager.GraphConstructorV2.Floats.Float import Float
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os



class DependencyGraphConstructor(Float):

    def __init__(self, texts: List[str], save_path: str, config: Config,anchor : Anchor
                load_preprocessed_data=False, naming_prefix='', node_name='dep' ,  start_data_load=0, end_data_load=-1):

        super(DependencyGraphConstructor, self)\
            .__init__(texts, save_path, config, anchor , load_preprocessed_data,
                      naming_prefix , node_name , start_data_load, end_data_load)
        self.dependencies = self.nlp.get_pipe("parser").labels
        self.settings = {"tokens_dep_weight" : 1,"dep_tokens_weight" : 1, "token_token_weight" : 2}
    def __find_dep_index(self , dependency : str):
        for dep_idx in range(len(self.dependencies)):
            if self.dependencies[dep_idx] == dependency:
                return dep_idx
        return -1 # means not found
    
    def __build_initial_dependency_vectors(self , dep_length : int):
        return torch.arange(0 , dep_length)
    
    def prepare_loaded_data(self , graph):
        graph = super().prepare_loaded_data(graph)
        graph[self.node_name].x = self.__build_initial_dependency_vectors(len(self.dependencies))
        return graph
    

    def add_nodes(self , doc , graph , use_compression=False):
        # nodes size is dependencies + tokens
        dep_length = len(self.dependencies)
        graph[self.node_name].length = dep_length
        if use_compression:
            graph[self.node_name].x = torch.full((dep_length,),-1, dtype=torch.float32)
        else:
            graph[self.node_name].x = self.__build_initial_dependency_vectors(dep_length)
        return graph
        
    def connect_nodes(self , graph , doc):
        token_dep_edge_index = []
        dep_token_edge_index = []
        token_dep_edge_attr = []
        dep_token_edge_attr = []
        for token in doc:
            if token.dep_ != 'ROOT':
                dep_idx = self.__find_dep_index(token.dep_)
                # not found protection
                if dep_idx != -1:
                    # edge from head token to dependency node
                    token_dep_edge_index.append([token.head.i , dep_idx])
                    token_dep_edge_attr.append(self.settings["tokens_dep_weight"])
                    # edge from dependency node to the token
                    dep_token_edge_index.append([dep_idx , token.i])
                    dep_token_edge_attr.append(self.settings["dep_tokens_weight"])
        graph[self.node_name , self.node_name + ' ' + self.anchor.node_name , self.anchor.node_name].edge_index = torch.transpose(torch.tensor(dep_token_edge_index, dtype=torch.int32) , 0 , 1) if len(dep_token_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        graph[self.anchor.node_name , self.anchor.node_name + ' ' + self.node_name , self.node_name].edge_index = torch.transpose(torch.tensor(token_dep_edge_index, dtype=torch.int32) , 0 , 1) if len(token_dep_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        graph[self.node_name , self.node_name + ' ' + self.anchor.node_name , self.anchor.node_name].edge_attr = torch.tensor(dep_token_edge_attr, dtype=torch.float32) 
        graph[self.anchor.node_name , self.anchor.node_name + ' ' + self.node_name , self.node_name].edge_attr = torch.tensor(token_dep_edge_attr, dtype=torch.float32)
        return graph
        

