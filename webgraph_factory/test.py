#from selenium import webdriver
#from selenium.webdriver.common.by import By

from web_graph import WebGraphFactory
from feature_extraction.extractor import FeatureExtractor
import pickle


def create_dataset(labeled_webgraphs):
    """
    webgraphs: list of tuples like [(webgraph1, label1), (webgraph2, label2)]
    """
    edges = []
    for graph_id, (webgraph, label) in enumerate(labeled_webgraphs):
        x, edge_index = webgraph.get_data()
        for edge in edge_index:
            edges.append((graph_id, edge[0], edge[1]))

    return edges


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

    print("Saving webgraphes into files...")
    with open("./dataset/graph_edges.csv", "wt") as fp_edges:
        with open("./dataset/graph_properties.csv", "wt") as fp_properties:
            with open("./dataset/node_features.csv", "wt") as fp_features:
                graph_id = 0
                webgraph.save_to_csv_files(graph_id, fp_edges, fp_properties, fp_features)

    dataset = create_dataset(labeled_webgraphs=[(webgraph, 'login')])

    with open("dump.pk", "wb") as fp: 
        pickle.dump(dataset, fp)