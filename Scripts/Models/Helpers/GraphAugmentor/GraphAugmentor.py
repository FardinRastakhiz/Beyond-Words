import uuid
from collections import OrderedDict

from Scripts.Models.Helpers.GraphLoader.GraphLoader import GraphLoader
from abc import ABC, abstractmethod


class GraphAugmentor(ABC):
    def __init__(self, name, inplace: bool = False):
        self.name = name
        self._unique_id = uuid.uuid4()
        self.inplace = inplace

    @abstractmethod
    def augment(self, graph_loader: GraphLoader):
        pass


