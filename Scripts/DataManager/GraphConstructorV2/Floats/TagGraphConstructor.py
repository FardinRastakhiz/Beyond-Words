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



class TagGraphConstructor(Float):

    def __init__(self, texts: List[str], save_path: str, config: Config,anchor : Anchor
                load_preprocessed_data=False, naming_prefix='', node_name='tag' ,  start_data_load=0, end_data_load=-1):

        super(TagGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, anchor , load_preprocessed_data,
                      naming_prefix , node_name , start_data_load, end_data_load)
        self.tags = self.nlp.get_pipe("tagger").labels
        self.settings = {"tokens_tag_weight" : 1, "token_token_weight" : 2}
    def __find_tag_index(self , tag : str):
        for tag_idx in range(len(self.tags)):
            if self.tags[tag_idx] == tag:
                return tag_idx
        return -1 # means not found
    
    def __build_initial_tag_vectors(self , tag_length : int):
        return torch.arange(0 , tag_length)
    
    def prepare_loaded_data(self , graph):
        graph = super().prepare_loaded_data(graph)
        graph[self.node_name].x = self.__build_initial_tag_vectors(len(self.tags))
        for t in graph.edge_types:
            if len(graph[t].edge_index) == 0:
                graph[i].edge_index = torch.empty(2, 0, dtype=torch.int32)
        return graph
    
    def _add_nodes(self , doc , graph , use_compression=True):
        tag_length = len(self.tags)
        graph[self.node_name].length = tag_length
        if use_compression:
            graph[self.node_name].x = torch.full((tag_length,),-1, dtype=torch.float32)
        else:
            graph[self.node_name].x = self.__build_initial_tag_vectors(tag_length)
        return graph
        
    def _connect_nodes(self , graph , doc):
        token_tag_edge_index = []
        tag_token_edge_index = []
        token_tag_edge_attr = []
        tag_token_edge_attr = []
        for token in doc:
            tag_idx = self.__find_tag_index(token.tag_)
            if tag_idx != -1:
                token_tag_edge_index.append([token.i , tag_idx])
                token_tag_edge_attr.append(self.settings["tokens_tag_weight"])
                tag_token_edge_index.append([tag_idx , token.i])
                tag_token_edge_attr.append(self.settings["tokens_tag_weight"])
        graph[self.node_name , self.node_name + ' ' + self.anchor.node_name , self.anchor.node_name].edge_index = torch.transpose(torch.tensor(tag_token_edge_index, dtype=torch.int32) , 0 , 1) if len(tag_token_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        graph[self.anchor.node_name , self.anchor.node_name + ' ' + self.node_name , self.node_name].edge_index = torch.transpose(torch.tensor(token_tag_edge_index, dtype=torch.int32) , 0 , 1) if len(token_tag_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        graph[self.node_name , self.node_name + ' ' + self.anchor.node_name , self.anchor.node_name].edge_attr = torch.tensor(tag_token_edge_attr, dtype=torch.float32) 
        graph[self.anchor.node_name , self.anchor.node_name + ' ' + self.node_name , self.node_name].edge_attr = torch.tensor(token_tag_edge_attr, dtype=torch.float32)
        return graph
        

