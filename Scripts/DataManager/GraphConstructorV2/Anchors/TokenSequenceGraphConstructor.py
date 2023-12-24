# Omid Davar @ 2023

import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructorV2.Anchors.Anchor import Anchor
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os
from abc import ABC, abstractmethod


class TokenSequenceGraphConstructor(Anchor):
    
    def __init__(self, texts: List[str], save_path: str, config: Config,
                load_preprocessed_data=False, naming_prefix='', node_name='token' , start_data_load=0, end_data_load=-1):

        super(AnchorGraphConstructor, self)\
            .__init__(texts, save_path, config, load_preprocessed_data,
                      naming_prefix, node_name , start_data_load, end_data_load)
            self.settings = {"token_token_weight" : 2}

    def prepare_loaded_data(self , graph):
        tokens = torch.zeros((len(graph[self.node_name].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
        for i in range(len(graph[self.node_name].x)):
            if graph[self.node_name].x[i] in self.nlp.vocab.vectors:
                tokens[i] = torch.tensor(self.nlp.vocab.vectors[graph[self.node_name].x[i]])
        graph[self.node_name].x = tokens
        return graph
    
    def _create_nodes(self , doc , use_compression=False):
        data = HeteroData()
        if use_compression:
            data[self.node_name].x = [-1 for i in range(len(doc))]
        else:
            data[self.node_name].x = torch.zeros((len(doc) , self.nlp.vocab.vectors_length), dtype=torch.float32)
        for token in doc:
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                if use_compression:
                    data[self.node_name].x[token.i] = token_id
                else:
                    data[self.node_name].x[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
        return data

    def connect_nodes(self , graph , doc):
        token_token_edge_index = []
        token_token_edge_attr = []
        for token in doc:
            # adding sequential edges between tokens 
            if token.i != len(doc) - 1:
                token_token_edge_index.append([token.i , token.i + 1])
                token_token_edge_attr.append(self.settings["token_token_weight"])
                token_token_edge_index.append([token.i + 1 , token.i])
                token_token_edge_attr.append(self.settings["token_token_weight"])
        graph[self.node_name , self.relation_name , self.node_name].edge_index = torch.transpose(torch.tensor(token_token_edge_index, dtype=torch.int32) , 0 , 1) if len(token_token_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        graph[self.node_name , self.relation_name , self.node_name].edge_attr = torch.tensor(token_token_edge_attr, dtype=torch.float32)
        return graph
        

