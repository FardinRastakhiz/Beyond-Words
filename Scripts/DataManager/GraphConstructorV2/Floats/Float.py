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
from abc import ABC, abstractmethod


class Float(GraphConstructor):

    def __init__(self, texts: List[str], save_path: str, config: Config,anchor : Anchor
                load_preprocessed_data=False, naming_prefix='', node_name='dep' ,  start_data_load=0, end_data_load=-1):

        super(Float, self)\
            .__init__(texts, self._Variables(), save_path, config, load_preprocessed_data,
                      naming_prefix , start_data_load, end_data_load)
        self.anchor = anchor
        self.node_name = node_name
        
    def to_graph(self, raw_data):
        doc = self.nlp(text)
        graph = self.anchor.to_graph(raw_data)
        data = self._add_nodes(doc , graph)
        return self._connect_nodes(data , doc)

    def to_graph_indexed(self, raw_data):
        doc = self.nlp(text)
        graph = self.anchor.to_graph_indexed(raw_data)
        data = self._add_nodes(doc , graph , True)
        return self._connect_nodes(data , doc)


    def prepare_loaded_graph(self , graph):
        graph = self.anchor.prepare_loaded_data(graph)
        return graph
    
    @abstractmethod
    def _add_nodes(self , doc , graph , use_compression=True):
        pass
    @abstractmethod
    def _connect_nodes(self , graph , doc):
        pass
        

