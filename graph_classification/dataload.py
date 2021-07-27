"""
See
https://docs.dgl.ai/en/0.6.x/new-tutorial/6_load_data.html
https://docs.dgl.ai/en/0.6.x/guide/data-dataset.html#guide-data-pipeline-dataset
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/tu_dataset.html

"""

import dgl
from dgl.data import DGLDataset
import torch
import os
import pickle

"""
The format of data files:

    graph_edges.csv: containing three columns:
        graph_id: the ID of the graph.
        src: the source node of an edge of the given graph.
        dst: the destination node of an edge of the given graph.

    graph_properties.csv: containing three columns:
        graph_id: the ID of the graph.
        label: the label of the graph.
        num_nodes: the number of nodes in the graph.

    node_features:
        graph_id: int
        node_id: int
        features: 1d-vector

"""

#edges = pd.read_csv('./graph_edges.csv')
#properties = pd.read_csv('./graph_properties.csv')
#edges.head()
#properties.head()


class WebGraphDataset(DGLDataset):
    """ A dataset for webgraph classification problem.
    Dataset should be a list of graphs and labels.
    getitem should return a pair (graph, label)
    """
    def __init__(self, path=None):
        super().__init__(name='web_graph')
        
        #with open(path) as fp:
        #    dataset = pickle.load(fp)
        #    print(dataset)

        fp = open("data/dataset.dump", encoding="utf8")
        dataset = pickle.load(fp)
        fp.close()

        
    def process(self):
        edges = pd.read_csv('./graph_edges.csv')
        properties = pd.read_csv('./graph_properties.csv')
        self.graphs = []
        self.labels = []
        
        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']
            
        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')
        
        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]
            
            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)
            
        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)
        
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)


if __name__ == "__main__":

    dataset = WebGraphDataset("data/dataset.dump")
    #graph, label = dataset[0]
    #print(graph, label)