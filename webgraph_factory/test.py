#from selenium import webdriver
#from selenium.webdriver.common.by import By

from web_graph import WebGraphFactory
from feature_extraction.extractor import FeatureExtractor
import pickle


def create_webgraphs(html_files):

    state_factory = WebGraphFactory(sequence_handler=None)
    FeatureExtractor.set_config(config=None)

    url = "https://demo1.testgold.dev/login"

    webgraphs = []

    for path in html_files:
        with open(path) as fp:
            html = fp.read()
        webgraph = state_factory.get_webgraph(html, url)
        webgraphs.append(webgraph)

    return webgraphs


def create_dataset(webgraphs, labels):
    """
    webgraphs: the list of webgraphs
    lables: the list of labels
    """
    edges = []
    properties = []
    features = []
    assert len(webgraphs) == len(labels)

    for graph_id, webgraph in enumerate(webgraphs):
        webgraph.visualize()
        x, edge_index = webgraph.get_data()
        num_nodes = len(x)
        label = labels[graph_id]
        properties.append((graph_id, label, num_nodes))
        for edge in edge_index:
            edges.append((graph_id, edge[0], edge[1]))
        for node_id, feature in enumerate(x):
            features.append((graph_id, node_id, feature))

    return {'edges': edges, 'properties': properties, 'features': features}


def create_dataset_as_list(webgraphs, labels):
    """
    webgraphs: the list of webgraphs
    lables: the list of labels
    """
    edges = []
    properties = []
    features = []
    assert len(webgraphs) == len(labels)

    dataset = []

    for graph_id, webgraph in enumerate(webgraphs):
        #webgraph.visualize()
        x, edge_index = webgraph.get_data()
        #num_nodes = len(x)
        #label = labels[graph_id]
        #properties.append((graph_id, label, num_nodes))
        #for edge in edge_index:
        #    edges.append((graph_id, edge[0], edge[1]))
        #for node_id, feature in enumerate(x):
        #    features.append((graph_id, node_id, feature))

        gr = {
            'id': graph_id, 
            'label': labels[graph_id],
            'num_nodes': len(x),
            'edges': edge_index,
            'x': x
        }
        dataset.append(gr)

    return dataset


if __name__ == "__main__":

    url = "https://demo1.testgold.dev/login"

    #options = webdriver.ChromeOptions()
    #options.add_argument('--ignore-certificate-errors')
    #options.add_argument("--test-type")
    #options.add_argument("--silent")
    #driver = webdriver.Chrome(chrome_options=options, executable_path='../chromedriver')
    #driver.get(url)
    #html = driver.page_source
    
    with open('init.html') as fp:
        html = fp.read()

    print("url:", url)
    print("html size:", len(html))

    state_factory = WebGraphFactory(sequence_handler=None)
    #html = self.get_html_code()
    webgraph = state_factory.get_webgraph(html, url)
    FeatureExtractor.set_config(config=None)
    print("webgraph:", webgraph)
    print("output:", webgraph.torch)
    print("type:", type(webgraph.torch))
    #webgraph.visualize()
    #x, edge_index = webgraph.get_data()
    #print(x)
    #print(edge_index)
    #webgraph.visualize()

    #print("Saving webgraphes into files...")
    #with open("./dataset/graph_edges.csv", "wt") as fp_edges:
    #    with open("./dataset/graph_properties.csv", "wt") as fp_properties:
    #        with open("./dataset/node_features.csv", "wt") as fp_features:
    #            graph_id = 0
    #            webgraph.save_to_csv_files(graph_id, fp_edges, fp_properties, fp_features)


    labels = ['login', 'other', 'other']
    html_files = ['html/0.html', 'html/1.html', 'html/2.html']

    webgraphs = create_webgraphs(html_files)
    #dataset = create_dataset(webgraphs, labels)
    dataset = create_dataset_as_list(webgraphs, labels)
    
    print(dataset)

    with open("dataset.dump", "wb") as fp: 
        pickle.dump(dataset, fp)