# Fardin Rastakhiz, Omid Davar @ 2023

import os
import pickle
from os import path
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, Any, List, Dict
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
from tqdm import tqdm
from Scripts.Configs.ConfigClass import Config
from enum import Enum
from flags import Flags
from Scripts.Utils.GraphUtilities import reweight_hetero_graph


class GraphConstructor(ABC):

    def __init__(self, raw_data, save_path: str, config: Config,
                 load_preprocessed_data: bool, naming_prefix: str = '', start_data_load=0, end_data_load=-1):
        
        self.raw_data = raw_data
        self.start_data_load = start_data_load
        self.end_data_load = end_data_load if end_data_load > 0 else len(self.raw_data)
        self.config: Config = config
        self.load_preprocessed_data = load_preprocessed_data
        self.save_path = os.path.join(config.root, save_path)
        self.naming_prefix = naming_prefix
        self.saving_batch_size = 1000 
        self._graphs: List = [None for r in range(end_data_load)]
        self.nlp = spacy.load(self.config.spacy.pipeline)

    def setup(self, load_preprocessed_data = True):
        self.load_preprocessed_data = True
        if load_preprocessed_data:
            for i in tqdm(range(self.start_data_load , self.end_data_load , self.saving_batch_size), desc ="Loding Graphs From File"):
                self.load_data_range(i , i + self.saving_batch_size)
        else:
            # save the content
            save_start = self.start_data_load
            for i in tqdm(range(self.start_data_load , self.end_data_load), desc =" Creating Graphs "):
                if i % self.saving_batch_size == 0:
                    if i != self.start_data_load: 
                        self.save_data_range(save_start, save_start + self.saving_batch_size)
                        save_start = i
            self.save_data_range(save_start, self.end_data_load)
            # Load the content
            # self._graphs: List = [None for r in range(self.end_data_load)]
            # self.setup(load_preprocessed_data=True)
            

    @abstractmethod
    def to_graph(self, raw_data):
        pass

    # below method returns torch geometric Data model with indexed nodes
    @abstractmethod
    def to_graph_indexed(self, raw_data):
        pass

    # below method gets graph loaded from indexed files and gives complete graph
    @abstractmethod
    def prepare_loaded_data(self , graph):
        pass

    def get_graph(self, idx: int):
        if self._graphs[idx] is None:
            self._graphs[idx] = self.to_graph(self.raw_data[idx])
        return self._graphs[idx]


    def get_graphs(self, ids: List | Tuple | range | np.array | torch.Tensor | any):
        not_loaded_ids = [idx for idx in ids if idx not in self._graphs]
        if len(not_loaded_ids) > 0 and self.load_preprocessed_data:
            self.load_data_list(not_loaded_ids)
        else:
            for idx in not_loaded_ids:
                self._graphs[idx] = self.to_graph(self.raw_data[idx])
        return {idx: self._graphs[idx] for idx in ids}

    def get_first(self):
        return self.get_graph(0)
    
    def draw_graph(self, idx: int):
        g = to_networkx(self.get_graph(idx), to_undirected=True)
        layout = nx.spring_layout(g)
        nx.draw(g, pos=layout)


    def save_data_range(self, start: int, end: int):
        data_list = []
        for i in range(start, end):
            data_list.append(self.to_graph_indexed(self.raw_data[i - self.start_data_load]))
        torch.save(data_list, path.join(self.save_path, f'{start}_{end}_compressed.pt'))
        
    def load_data_range(self, start: int, end: int):
        data_list = torch.load(path.join(self.save_path, f'{start}_{end}_compressed.pt'))
        index = 0
        for i in range(start, end):
            self._graphs[i - self.start_data_load] = self.prepare_loaded_data(data_list[index])
            index += 1
        pass    

        
    def reweight(self, idx : int , triplet : tuple , weight):
        is_available = isinstance(self._graphs[idx] , HeteroData)
        if is_available:
            return reweight_hetero_graph(self._graphs[idx] , triplet , weight)
        else:
            return None
        
    def reweight_all(self , triplet : tuple , weight):
        for i in range(len(self._graphs)):
            self._graphs[i] = self.reweight(i , triplet , weight)
        
            
            
        
    
