import torch
from torch_geometric.data import Data

import bs4
from bs4 import BeautifulSoup
from lxml import etree
import networkx as nx
import numpy as np

from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import hashlib
import html_similarity
import pickle

from src.utils.html_utils import classify_html
from src.utils.general_utils import lxml2xpath
from src.utils.similarity import calculate_elements_similarity
from src.config import config

from .element import PageElement

def _get_similarity_func(similarity_type):
    if similarity_type == 'structural':
        return html_similarity.structural_similarity
    if similarity_type == 'style':
        return html_similarity.style_similarity
    if similarity_type == 'both':
        return html_similarity.similarity
    raise NotImplementedError(f'{similarity_type} type is not supported!')


SKIP_TAGS = ['script','noscript']

class WebGraphFactory():
    def __init__(self,sequence_handler,similarity_thresold=0.95,similarity_type='both'):
        self._graphs = []
        self._prev_graph = None
        self._get_similarity = _get_similarity_func(similarity_type)
        self._threshold = similarity_thresold

        self.sequence_handler = sequence_handler

        self.global_interactables = []

    @property
    def state_amount(self):
        return len(self._graphs)

    def get_state(self,id):
        return self._graphs[id]

    def get_webgraph(self,html,url):
        new_graph = None
        for graph in self._graphs:
            if graph.url == url:
                sim = self._get_similarity(graph.html,html)
                if sim >= self._threshold:       
                    new_graph = graph                    
                    break
        if new_graph:
            if self._prev_graph and self._prev_graph.url != url:
                new_graph.refresh()
        else:
            new_graph = self.create_graph(html,url)
            self._graphs.append(new_graph)
        self._prev_graph = new_graph
        return new_graph

    def load_pickled_states(self,path):
        pkls = list(path.glob('*.pkl'))
        self._graphs = []
        state_map = {}
        states = []
        for state_pkl in pkls:
            with open(state_pkl,'rb') as f:
                state = pickle.load(f)
            states.append(state)
            state_map[state_pkl.name] = state
        
        state2name = {v:k for k, v in state_map.items()}

        for s in sorted(states,key=lambda x: x.global_id):
            append = True
            for graph in self._graphs:
                sim = self._get_similarity(graph.html,s.html)
                if sim >= self._threshold:  
                    name = state2name[s]
                    state_map[name] = graph
                    append = False
                    break
            if append:
                s.global_id = len(self._graphs)
                self._graphs.append(s)
        return state_map


    def create_graph(self,html,url,skip_tags=SKIP_TAGS):
        self._current_html = html
        self._current_url = url
        types_dict = config.externals.parser_module.parse(html)
        body = BeautifulSoup(html, 'html.parser').body
        element_id = 0
        element = PageElement.from_bs4_object(element_id, body,types_dict)
        edges = []
        nodes = [element]
        interactables = {}
        self._traverse(body,element_id,edges,nodes,interactables,types_dict,skip_tags=skip_tags)
        parsed_html = edges,nodes,interactables
        return WebGraph(len(self._graphs), html,url,parsed_html)

    def _traverse(self,parent,element_id,edges,nodes,interactables,types_dict,skip_tags=SKIP_TAGS):
        parent_id = element_id
        for child in parent.children:
            if isinstance(child,bs4.element.NavigableString):
                continue
            if str(child.name) in skip_tags:
                continue
            element = PageElement.from_bs4_object(len(nodes), child,types_dict)
            if element.interaction_type != 'none':
                buckets = self._get_bucket_amount(element)
                interactables[len(nodes)] = buckets 
            
            edges.append( [parent_id,len(nodes)] )
            edges.append( [len(nodes),parent_id] )
            nodes.append(element)
            self._traverse(child,len(nodes)-1,edges,nodes,interactables,types_dict,skip_tags)

    def _get_bucket_amount(self,element):

        buckets = 1
        if element.interaction_type == 'selectable':
            buckets = len(element.options)
        elif element.interaction_type == 'enterable': 
            body = etree.HTML(self._current_html)
            selectors_dict = {'xpath':body.xpath, 'css selector':body.cssselect}
            for url,(selector_method,selector_value),interval_len in self.sequence_handler.steps_intervals:
                find = selectors_dict[selector_method]
                # if url == self._current_url: # TODO: issue #11
                elem = find(selector_value)
                if len(elem) > 0:
                    elem = elem[0]
                    if element.xpath == body.getroottree().getpath(elem):
                        buckets = interval_len
                        break    
        buckets = min(buckets,config.env.discretization)
        return buckets
    

class WebGraph():
    def __init__(self, global_id,html, url, parsed_html, use_dependency=False):

        self.global_id = global_id
        self.url = url

        self.html = html
        self._bs4_body = None
        self._etree = None

        self.edge_index,self.nodes,self.interactables = parsed_html

        self._nx_graph = None
        self._torch_graph = None

        # self.dependency_dict = None
        # if use_dependency:
        #     self.dependency_dict = self.create_dependency_dict()

        self._similarity_dict = {}

    @property
    def torch(self):
        if not self._torch_graph:
            edge_index = torch.tensor(self.edge_index,dtype = torch.long)
            x = [e.get_features() for e in self.nodes]
            x = torch.tensor(x,dtype = torch.float)
            self._torch_graph = Data(x=x, edge_index=edge_index.t().contiguous())
        return self._torch_graph

    @property
    def nx(self):
        if not self._nx_graph:
            self._nx_graph = nx.Graph()
            nodes = [(node.id,node.__dict__) for node in self.nodes]
            self._nx_graph.add_nodes_from(nodes)
            self._nx_graph.add_edges_from(self.edge_index)
        return self._nx_graph

    @property
    def lxml_body(self):
        if not self._etree:
           self._etree = etree.HTML(str(self.bs4_body))
        return self._etree

    @property
    def bs4_body(self):
        if not self._bs4_body:
           self._bs4_body = BeautifulSoup(self.html, 'lxml').body
        return self._bs4_body

    def refresh(self):
        for node in self.nodes:
            node._available = True

    def create_dependency_dict(self):
        dependency_dict = {}
        classifed_forms = classify_html( str(self._bs4_body))
        for form in classifed_forms:
            if form['form'] in ['login','registration']:
                fields = form['fields']
                if 'submit button' in fields:
                    button_node = self.find_node_by_name(fields['submit button']) 
                    if button_node:
                        pswd_name = fields.get('password',None)
                        login_name = fields.get('username',False) or fields.get('email',False) or fields.get('username or email',False)
                        pswd_node = self.find_node_by_name(pswd_name)
                        login_node = self.find_node_by_name(login_name)
                        if pswd_node and login_node:
                            dependency_dict[button_node] = [login_node,pswd_node]
                            button_node._available = False
        return dependency_dict

    def get_first_interactable_parent(self,element,limit = 10):
        
        for i,parent_id in enumerate(nx.dfs_preorder_nodes(self.nx,element.id)):
            if i == limit or parent_id == 0:
                break

            elem = self.get_element(parent_id)
            if elem.interaction_type != 'none':
                return elem
        return None



    def find_node_by_name(self,name):
        if not name:
            return None
        split = name.split('[')
        name = split[0]
        idx = 0 if len(split) == 1 else int(split[1][:-1])
        found = 0
        for node in self.nodes:
            if node.name == name or node.tag == name:
                if found == idx:
                    return node
                found +=1
        return None

    def find_node_by_selector(self,selector_method,selector_value):
        selectors_dict = {'xpath':self.lxml_body.xpath, 
                          'css selector':self.lxml_body.cssselect}
        find = selectors_dict[selector_method]
        elem_to_find = find(selector_value)
        if len(elem_to_find) > 0:
            elem_to_find = elem_to_find[0]
            for node in self.nodes:
                if node.tag == elem_to_find.tag and node.xpath == self.lxml_body.getroottree().getpath(elem_to_find):            
                    return node
        return None

    def get_element(self,id):
        element = self.nodes[id]
        # if element.tag == 'input':
        #     element._available = False # Input element can be interacted only once. ### TODO: It's not always true!
        #     if self.dependency_dict:
        #         for k,v in self.dependency_dict.items():
        #             if all([not e.available for e in v]):
        #                 k.available = True
        return element

    def get_similar_elements(self,driver, element=None,node_id=None,threshold=0.8):
        assert element or node_id, 'Similar objects are found based on element object or node id.'
        if not element:
            element = self.nodes[node_id]

        if element not in self._similarity_dict:
            window_size = driver.get_window_size()
            interaction_type = element.interaction_type
            similar = []
            for node_id in self.interactables.keys():
                node = self.nodes[node_id]
                if node.interaction_type == interaction_type:
                    score = calculate_elements_similarity(element,node,driver,window_size)
                    if score > threshold:
                        similar.append( (node,score) )
            similar = sorted(similar,key=lambda x: x[1],reverse=True)
            self._similarity_dict[element] = similar
        return self._similarity_dict[element]

    def get_feature_len(self):
        return len(self.nodes[0].get_features())

    def visualize(self):
        nx_graph = self.nx
        tags = nx.get_node_attributes(nx_graph, 'tag')
        pos = graphviz_layout(nx_graph, prog='dot')
        nx.draw(nx_graph, labels=tags,pos=pos)   
        plt.show()

    def __len__(self):
        return len(self.nodes)

    def __hash__(self):
        return int(hashlib.md5(str(self.bs4_body).encode()).hexdigest(),16)

    def __eq__(self, other):
        return isinstance(other, WebGraph) and self.bs4_body == other.bs4_body

    def __ne__(self, other):
        return not(self == other)


    def _pickle_value(self,k,v):
        if k in ['_etree','_nx_graph','_torch_graph','_bs4_body']:
            return None
        return v

    def __getstate__(self):
        return {k:self._pickle_value(k,v) for (k, v) in self.__dict__.items() }

