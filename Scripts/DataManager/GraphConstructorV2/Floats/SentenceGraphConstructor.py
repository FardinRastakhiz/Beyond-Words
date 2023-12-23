# Omid Davar @ 2023


import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructorV2.GraphConstructor import GraphConstructor
from Scripts.DataManager.GraphConstructorV2.Anchors.Anchor import Anchor
from Scripts.DataManager.GraphConstructorV2.Floats.Float import Float
from torch_geometric.graph import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os



class SentenceGraphConstructor(Float):

    def __init__(self, texts: List[str], save_path: str, config: Config,anchor : Anchor
                load_preprocessed_graph=False, naming_prefix='', node_name=self.node_name ,  start_graph_load=0, end_graph_load=-1):

        super(SentenceGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, anchor , load_preprocessed_graph,
                      naming_prefix , node_name , start_graph_load, end_graph_load)
        self.settings = {"token_sentence_weight" : 1}
    
    def _add_nodes(self , doc , graph , use_compression=True):
        sentence_embeddings = [sent.vector for sent in doc.sents]
        graph[self.node_name].x = torch.tensor(sentence_embeddings, dtype=torch.float32)
        return graph
    
    def _connect_nodes(self , graph , doc):
        sentence_token_edge_index = []
        token_sentence_edge_index = []
        sentence_token_edge_attr = []
        token_sentence_edge_attr = []
        sent_index = -1
        for token in doc:
            # connecting tokens to sentences
            if token.is_sent_start:
                sent_index += 1
            token_sentence_edge_index.append([token.i, sent_index])
            sentence_token_edge_index.append([sent_index, token.i])
            token_sentence_edge_attr.append(self.settings['token_sentence_weight'])
            sentence_token_edge_attr.append(self.settings['token_sentence_weight'])
        data[self.anchor.node_name , self.anchor.node_name + ' ' + self.node_name , self.node_name].edge_index = torch.transpose(torch.tensor(token_sentence_edge_index, dtype=torch.int32) , 0 , 1) if len(token_sentence_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data[self.node_name , self.node_name + ' ' + self.anchor.node_name , self.anchor.node_name].edge_index = torch.transpose(torch.tensor(sentence_token_edge_index, dtype=torch.int32) , 0 , 1) if len(sentence_token_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data[self.anchor.node_name , self.anchor.node_name + ' ' + self.node_name , self.node_name].edge_attr = torch.tensor(token_sentence_edge_attr, dtype=torch.float32)
        data[self.node_name , self.node_name + ' ' + self.anchor.node_name , self.anchor.node_name].edge_attr = torch.tensor(sentence_token_edge_attr, dtype=torch.float32)
        return data

        

