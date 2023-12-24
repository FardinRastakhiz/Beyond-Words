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

from spacytextblob.spacytextblob import SpacyTextBlob



class SentimentGraphConstructor(Float):

    def __init__(self, texts: List[str], save_path: str, config: Config,anchor : Anchor
                load_preprocessed_graph=False, naming_prefix='', node_name=self.node_name ,  start_graph_load=0, end_graph_load=-1):

        super(SentimentGraphConstructor, self)\
            .__init__(texts, save_path, config , anchor, load_preprocessed_graph,
                      naming_prefix , node_name , start_graph_load, end_graph_load)
    

    def _build_initial_sentiment_vector(self):
        return torch.zeros((2 , self.nlp.vocab.vectors_length), dtype=torch.float32)  
    
    def prepare_loaded_graph(self , graph):
        graph = super().prepare_loaded_data(graph)
        graph['sentiment'].x = self._build_initial_sentiment_vector()
        return graph
    
    def add_nodes(self , doc , graph , use_compression=False):
        graph['sentiment'].x = self._build_initial_sentiment_vector()
        return graph
    def connect_nodes(self , graph , doc):
        sentiment_token_edge_index = []
        token_sentiment_edge_index = []
        sentiment_token_edge_attr = []
        token_sentiment_edge_attr = []
        for token in doc:
            if token._.blob.polarity > 0:
                token_sentiment_edge_index.append([token.i, 1])
                sentiment_token_edge_index.append([1, token.i])
                token_sentiment_edge_attr.append(abs(token._.blob.polarity))
                sentiment_token_edge_attr.append(abs(token._.blob.polarity))
            if token._.blob.polarity < 0:
                token_sentiment_edge_index.append([token.i, 0])
                sentiment_token_edge_index.append([0, token.i])
                token_sentiment_edge_attr.append(abs(token._.blob.polarity))
                sentiment_token_edge_attr.append(abs(token._.blob.polarity))
        data[self.anchor.node_name , self.anchor.node_name + ' ' + self.node_name , self.node_name].edge_index = torch.transpose(torch.tensor(token_sentiment_edge_index, dtype=torch.int32) , 0 , 1) if len(token_sentiment_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data[self.node_name , self.node_name + ' ' + self.anchor.node_name , self.anchor.node_name].edge_index = torch.transpose(torch.tensor(sentiment_token_edge_index, dtype=torch.int32) , 0 , 1) if len(sentiment_token_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data[self.anchor.node_name , self.anchor.node_name + ' ' + self.node_name , self.node_name].edge_attr = torch.tensor(token_sentiment_edge_attr, dtype=torch.float32)
        data[self.node_name , self.node_name + ' ' + self.anchor.node_name , self.anchor.node_name].edge_attr = torch.tensor(sentiment_token_edge_attr, dtype=torch.float32)
        return data

        

