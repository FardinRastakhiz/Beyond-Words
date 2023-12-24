# Omid Davar @ 2023

import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructorV2.GraphConstructor import GraphConstructor
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os
from abc import ABC, abstractmethod
from enum import Enum
class AnchorType(Enum):
    TOKENSEQUENCE = 1
    
    @staticmethod
    get_anchor_by_type(anchorType , texts , save_path, config):
        if anchorType == AnchorType.TOKENSEQUENCE:
            return TokenSequenceGraphConstructor(texts , save_path, config)

class AnchorGraphConstructor(GraphConstructor):
    
    def __init__(self, texts: List[str], save_path: str, config: Config,
                load_preprocessed_data=False, naming_prefix='' , node_name = 'token', start_data_load=0, end_data_load=-1):

        super(AnchorGraphConstructor, self)\
            .__init__(texts, save_path, config, load_preprocessed_data,
                      naming_prefix , start_data_load, end_data_load)
        self.node_name = node_name
        self.relation_name = self.node_name + '_' + self.node_name
    def to_graph(self, raw_data):
        doc = self.nlp(text)
        data = self._create_nodes(doc)
        return self.connect_nodes(data , doc)

    def to_graph_indexed(self, raw_data):
        doc = self.nlp(text)
        data = self._create_nodes(doc , True)
        return self.connect_nodes(data , doc)

    # below method gets graph loaded from indexed files and gives complete graph
    @abstractmethod
    def prepare_loaded_data(self , graph):
        pass
    
    @abstractmethod
    def _create_nodes(self , doc , use_compression=False):
        pass
    @abstractmethod
    def connect_nodes(self , graph , doc):
        pass
        

