import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os


class SequentialGraphConstructor(GraphConstructor):
    
    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(SequentialGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''
    def __init__(self, texts: List[str], save_path: str, config: Config,
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend='', use_general_node=False):

        super(SequentialGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, lazy_construction, load_preprocessed_data,
                      naming_prepend)
        self.settings = {"tokens_general_weight" : 1, "token_token_weight" : 2 , "general_tokens_weight" : 2}
        self.use_general_node = use_general_node
        if self.load_preprocessed_data:
            if not self.lazy_construction:
                self.load_all_data()
            else:
                self.load_var()
        else:
            self.var.nlp_pipeline = self.config.spacy.pipeline
            self.var.graph_num = len(self.raw_data)
            self.nlp = spacy.load(self.var.nlp_pipeline)
            if not self.lazy_construction:
                for i in range(len(self.raw_data)):
                    if i not in self._graphs:
                        if i % 100 == 0:
                            print(f'i: {i}')
                        self._graphs[i] = self.to_graph(self.raw_data[i])
                        self.var.graphs_name[i] = f'{self.naming_prepend}_{i}'
            self.save_all_data()

    def to_graph(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        return self.__create_graph(doc)
    def __create_graph(self , doc):
        docs_length = len(doc)
        if self.use_general_node:
            docs_length += 1
        node_attr = torch.zeros((docs_length, self.nlp.vocab.vectors_length), dtype=torch.float32)
        edge_index = []
        edge_attr = []
        
        # for idx in range(tags_length):
            # if vevtorizing of dependencies is needed, do it here
            # node_attr[idx] = sth ...
        for token in doc:
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                if self.use_general_node:
                    node_attr[token.i + 1] = torch.tensor(self.nlp.vocab.vectors[token_id])
                else:
                    node_attr[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
            if self.use_general_node:
                edge_index.append([token.i + 1 , 0])
                edge_attr.append(self.settings['tokens_general_weight'])
                edge_index.append([0 , token.i + 1])
                edge_attr.append(self.settings['general_tokens_weight'])
            # adding sequential edges between tokens - uncomment the codes for vectorized edges
            if token.i != len(doc) - 1:
                # using zero vectors for edge features
                if self.use_general_node:
                    edge_index.append([token.i + 1 , token.i + 2])
                    edge_index.append([token.i + 2 , token.i + 1])
                    # self.edge_attr.append(torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32))
                    edge_attr.append(self.settings["token_token_weight"]) 
                    edge_attr.append(self.settings["token_token_weight"]) 
                else:
                    edge_index.append([token.i , token.i + 1])
                    edge_index.append([token.i + 1 , token.i])
                    # self.edge_attr.append(torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32))
                    edge_attr.append(self.settings["token_token_weight"]) 
                    edge_attr.append(self.settings["token_token_weight"]) 
        # self.node_attr = node_attr
        edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.long) , 0 , 1)
        # self.edge_attr = edge_attr # vectorized edge attributes
        return Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr)
    def draw_graph(self , idx : int):
        node_tokens = []
        if self.use_general_node:
            node_tokens.append("gen_node")
        doc = self.nlp(self.raw_data[idx])
        for t in doc:
            node_tokens.append(t.lemma_)
        graph_data = self.get_graph(idx)
        g = to_networkx(graph_data)
        layout = nx.spring_layout(g)
        nx.draw(g, pos=layout)
        words_dict = {i: node_tokens[i] for i in range(len(node_tokens))}
        # edge_labels_dict = {(graph_data.edge_index[0][i].item() , graph_data.edge_index[1][i].item()) : { "dep" : graph_data.edge_attr[i]} for i in range(len(graph_data.edge_attr))}
        # nx.set_edge_attributes(g , edge_labels_dict)
        nx.draw_networkx_labels(g, pos=layout, labels=words_dict)
        # nx.draw_networkx_edge_labels(g, pos=layout)
    def to_graph_indexed(self, text: str):
        # TODO : implement this
        pass
    def convert_indexed_nodes_to_vector_nodes(self, graph):
        # TODO : implement this
        pass
        

