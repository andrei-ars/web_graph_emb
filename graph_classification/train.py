import sys
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from dataload_data import WebGraphDataset
from graph_classification_model import GraphClassificationModel

if __name__ == "__main__":

    cl_model = GraphClassificationModel()
    dataset = WebGraphDataset("dataset/dataset.dump")
    cl_model.train(dataset, num_epochs=21)