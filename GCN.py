import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super().__init__()
        self.dataset = dataset
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(self.dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, self.dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
