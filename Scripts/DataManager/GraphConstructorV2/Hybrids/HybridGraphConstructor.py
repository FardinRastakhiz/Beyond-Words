# Omid Davar @ 2023



from Scripts.DataManager.GraphConstructorV2.GraphConstructor import GraphConstructor
from Scripts.DataManager.GraphConstructorV2.Anchors.Anchor import Anchor
from Scripts.DataManager.GraphConstructorV2.Anchors.Anchor import AnchorType
from Scripts.DataManager.GraphConstructorV2.Floats.Float import Float
from Scripts.DataManager.GraphConstructorV2.Floats.Float import FloatType
from Scripts.DataManager.GraphConstructorV2.Connectors.VirtualConnector import VirtualConnector
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os


class HybridGraphConstructor(GraphConstructor):

    def __init__(self, texts: List[str], save_path: str, config: Config,anchor : AnchorType , floats : FloatType[] , connectors : VirtualConnector[],
                load_preprocessed_data=False, naming_prefix='' ,  start_data_load=0, end_data_load=-1):

        super(Float, self)\
            .__init__(texts, save_path, config, load_preprocessed_data,
                      naming_prefix , start_data_load, end_data_load)
        self.anchor = AnchorType.get_anchor_by_type(anchor , texts , save_path, config)
        for f in floats:
            self.floats.append(FloatType.get_float_by_type(f , texts , save_path, config , self.anchor))
        self.connectors = connectors
        
        
    def to_graph(self, raw_data):
        doc = self.nlp(text)
        graph = self.anchor.to_graph(raw_data)
        for f in self.floats:
            graph = f.add_nodes(doc , graph , False)
            graph = f.connect_nodes(graph , doc)
        for c in connectors:
            graph = c.connect(graph)
        return graph

    def to_graph_indexed(self, raw_data):
        doc = self.nlp(text)
        graph = self.anchor.to_graph_indexed(raw_data)
        for f in self.floats:
            graph = f.add_nodes(doc , graph , True)
            graph = f.connect_nodes(graph , doc)
        for c in connectors:
            graph = c.connect(graph)
        return graph

    def prepare_loaded_graph(self , graph):
        graph = self.anchor.prepare_loaded_data(graph)
        for f in self.floats:
            graph = f.prepare_loaded_data(graph)
        return graph
    
        