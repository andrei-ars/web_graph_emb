import sys
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from dataload_data import WebGraphDataset
from graph_classification_model import GraphClassificationModel

if __name__ == "__main__":

    dataset = WebGraphDataset("dataset/dataset.dump")
    cl_model = GraphClassificationModel(dataset)
    cl_model.train(batch_size=2, num_epochs=30)