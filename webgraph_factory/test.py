#from selenium import webdriver
#from selenium.webdriver.common.by import By

from web_graph import WebGraphFactory
from feature_extraction.extractor import FeatureExtractor

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
    x, edge_index = webgraph.get_data()
    print(x)
    print(edge_index)
    webgraph.visualize()