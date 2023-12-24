# Omid Davar @ 2023


import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructorV2.GraphConstructor import GraphConstructor
from Scripts.DataManager.GraphConstructorV2.Anchors.Anchor import Anchor
from Scripts.DataManager.GraphConstructorV2.Anchors.Anchor import AnchorType
from Scripts.DataManager.GraphConstructorV2.Anchors.TokenSequenceGraphConstructor import TokenSequenceGraphConstructor
from Scripts.DataManager.GraphConstructorV2.Floats.DependencyGraphConstructor import DependencyGraphConstructor
from Scripts.DataManager.GraphConstructorV2.Floats.TagGraphConstructor import TagGraphConstructor
from Scripts.DataManager.GraphConstructorV2.Floats.SentimentGraphConstructor import SentimentGraphConstructor
from Scripts.DataManager.GraphConstructorV2.Floats.SentenceGraphConstructor import SentenceGraphConstructor
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os
from abc import ABC, abstractmethod
from enum import Enum
class FloatType(Enum):
    DEPENDENCY = 1
    SENTENCE = 2
    SENTIMENT = 3
    TAG = 4
    
    @staticmethod
    get_float_by_type(floatType , texts , save_path, config , anchor = None):
        if floatType == FloatType.DEPENDENCY:
            return DependencyGraphConstructor(texts , save_path, config , anchor if anchor is not None else AnchorType.TOKENSEQUENCE)
        if floatType == FloatType.SENTENCE:
            return SentenceGraphConstructor(texts , save_path, config , anchor if anchor is not None else AnchorType.TOKENSEQUENCE)
        if floatType == FloatType.SENTIMENT:
            return SentimentGraphConstructor(texts , save_path, config , anchor if anchor is not None else AnchorType.TOKENSEQUENCE)
        if floatType == FloatType.TAG:
            return TagGraphConstructor(texts , save_path, config , anchor if anchor is not None else AnchorType.TOKENSEQUENCE)

class Float(GraphConstructor):

    def __init__(self, texts: List[str], save_path: str, config: Config,anchor : AnchorType
                load_preprocessed_data=False, naming_prefix='', node_name='dep' ,  start_data_load=0, end_data_load=-1):

        super(Float, self)\
            .__init__(texts, save_path, config, load_preprocessed_data,
                      naming_prefix , start_data_load, end_data_load)
        self.anchor = AnchorType.get_anchor_by_type(anchor , texts , save_path, config)
        self.node_name = node_name
        
    def to_graph(self, raw_data):
        doc = self.nlp(text)
        graph = self.anchor.to_graph(raw_data)
        data = self.add_nodes(doc , graph)
        return self.connect_nodes(data , doc)

    def to_graph_indexed(self, raw_data):
        doc = self.nlp(text)
        graph = self.anchor.to_graph_indexed(raw_data)
        data = self.add_nodes(doc , graph , True)
        return self.connect_nodes(data , doc)


    def prepare_loaded_graph(self , graph):
        graph = self.anchor.prepare_loaded_data(graph)
        return graph
    
    @abstractmethod
    def add_nodes(self , doc , graph , use_compression=False):
        pass
    @abstractmethod
    def connect_nodes(self , graph , doc):
        pass
        

