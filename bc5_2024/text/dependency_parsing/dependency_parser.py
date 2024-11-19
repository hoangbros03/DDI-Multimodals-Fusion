import networkx as nx
import spacy
from tqdm import tqdm

from bc5_2024.utils import offset_to_idx

class DependencyParser:
    def __init__(self, nlp):
        self.nlp = nlp
    
    def get_dependency_graph(self, text):
        doc = self.nlp(text)
        edges = list()
        for token in doc:
            for child in token.children:
                edges.append((token.i, child.i))
        G = nx.Graph(edges)

        return G
    
    def render(self, text):
        doc = self.nlp(text)
        spacy.displacy.render(doc, style="dep", jupyter=True, options={"compact": True})
    
    def get_sdp(self,
                text,
                source_i,
                target_i):
        graph = self.get_dependency_graph(text)
        source = source_i
        target = target_i

        return nx.shortest_path(graph, source=source, target=target)

    def get_edge_dep(self, text, source_i, target_i):
        doc = self.nlp(text)
        direction = None

        for token in doc:
            if token.i == source_i:
                if token.head.i == target_i:
                    direction = 'reverse'
                    return direction, token.dep_
                else:
                    for child in token.children:
                        if child.i == target_i:
                            direction = 'forward'
                            return direction, child.dep_

        return None
    
    def get_sdp_with_dep(self,
                         text,
                         source_i,
                         target_i):
        path_with_dep = list()

        path = self.get_sdp(text,
                            source_i,
                            target_i)

        for i in range(len(path) - 1):
            edge_dep_tmp = self.get_edge_dep(text, path[i], path[i+1])
            path_with_dep.append((path[i], edge_dep_tmp, path[i+1]))

        return path_with_dep
    
    def get_sdp_one(self, c):
        source_i = offset_to_idx(c['text'], c['e1']['@charOffset'], self.nlp)[0]
        target_i = offset_to_idx(c['text'], c['e2']['@charOffset'], self.nlp)[0]
        assert isinstance(source_i, int) and isinstance(target_i, int)
        sdp = self.get_sdp_with_dep(c['text'], source_i, target_i)

        return sdp
    
    def get_sdp_all(self, all_candidates):
        sdp_list = list()
        for c in tqdm(all_candidates):
            sdp_list.append(self.get_sdp_one(c))

        return sdp_list