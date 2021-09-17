import sys
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from dataload_data import WebGraphDataset

#WebGraphDataset("data/dataset.dump")

#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#dataset = dataset.shuffle()

#dataset = WebGraphDataset("data/dataset.dump")
dataset = WebGraphDataset()
train_dataset = dataset[:2]
test_dataset = dataset[2:]

print(dataset[0])
print(type(dataset[0]))
#sys.exit()

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):

    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = x.float() # TO FLOAT!

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


class Graph_Classification_Model:

    def __init__(self):

        self.model = GCN(hidden_channels=64)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, num_epochs=100):
        for epoch in range(1, num_epochs):
            self.train_epoch()
            train_acc = self.test(train_loader)
            test_acc = self.test(test_loader)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    def train_epoch(self):
        self.model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            print(data)
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    def test(self, loader):
        print("try eval")
        self.model.eval()
        print("ok")

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            print("data.x:", data.x)
            print("data.edge_index:", data.edge_index)
            print("data.batch:", data.batch)
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == "__main__":

    cl_model =  Graph_Classification_Model()
    cl_model.train(num_epochs=21)