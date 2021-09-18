import sys
import random
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from dataload_data import WebGraphDataset

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):

    def __init__(self, dataset, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = x.float() # to float!

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


class GraphClassificationModel:

    def __init__(self, dataset):

        self.dataset = dataset
        self.model = GCN(self.dataset, hidden_channels=64)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, batch_size, num_epochs=100, valid_percent=0.3):

        dataset = self.dataset
        #random.shuffle(dataset)
        valid_size = int(valid_percent * len(dataset))
        train_size = len(dataset) - valid_size

        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        print("dataset size:", len(dataset))
        print("train_dataset:", len(train_dataset))
        print("test_dataset:", len(test_dataset))

        print(dataset[0])
        print(type(dataset[0]))
        #sys.exit()
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(1, num_epochs):
            self.train_epoch(self.train_loader)
            train_acc = self.test(self.train_loader)
            test_acc = self.test(self.test_loader)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    def train_epoch(self, loader):
        self.model.train()

        for data in loader:  # Iterate in batches over the training dataset.
            print(data)
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    def test(self, loader):
        print("try eval")
        self.model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            #print("data.x:", data.x)
            #print("data.edge_index:", data.edge_index)
            #print("data.batch:", data.batch)
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == "__main__":

    dataset = WebGraphDataset("dataset/dataset.dump")
    cl_model = GraphClassificationModel(dataset)
    cl_model.train(batch_size=1, num_epochs=10)