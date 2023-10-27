
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from Scripts.DataManager.GraphLoader.GraphLoader import GraphLoader
from torch.utils.data.dataset import random_split
import torch


class GraphConstructorDataset(Dataset):

    def __init__(self, graph_constructor: GraphConstructor, graph_labels):
        self.graph_constructor = graph_constructor
        self.graph_labels = graph_labels

    def __getitem__(self, index):
        x = self.graph_constructor.get_graph(index)
        y = self.graph_labels[index]
        return x, y

    def __len__(self):
        return self.graph_constructor.var.graph_num


class GLabeledGraphLoader(GraphLoader):

    def __init__(self, graph_constructor: GraphConstructor, graph_label, batch_size, device,
                 test_size=0.2, val_size=0.15, num_workers=2, drop_last=True, *args, **kwargs):
        self.device = device
        self.batch_size = batch_size
        sample_graph = graph_constructor.get_first()
        self.num_node_features = sample_graph.num_features
        self.num_classes = len(torch.unique(graph_label))
        super(GLabeledGraphLoader, self)\
            .__init__(device, test_size, val_size, *args, **kwargs)

        self.dataset = GraphConstructorDataset(graph_constructor, graph_label)
        self.__train_dataset, self.__val_dataset, self.__test_dataset =\
            random_split(self.dataset, [1-val_size-test_size, val_size, test_size])
        self.__train_dataloader = DataLoader(self.__train_dataset, batch_size=batch_size, drop_last=drop_last,
                                             shuffle=True, num_workers=num_workers)
        self.__val_dataloader = DataLoader(self.__val_dataset, batch_size=batch_size, drop_last=drop_last,
                                           shuffle=True, num_workers=num_workers)
        self.__test_dataloader = DataLoader(self.__test_dataset, batch_size=batch_size, drop_last=drop_last,
                                            shuffle=True, num_workers=num_workers)

    def get_train_data(self):
        return self.__train_dataloader

    def get_test_data(self):
        return self.__test_dataloader

    def get_val_data(self):
        return self.__val_dataloader

