from igraph import *

class DocumentGraph:
    def __init__(self):
        self.obj_graph = None
        self.commnity = None
        self.list_community = None
        
        self.list_keyword2freq_insubtopics = []
        self.list_edge2freq_insubtopics = []
        self.org_subtopic_size = 0
        self.cut_subtopic_size = 0
        
    # input : 옵션, id에서 키워드로 사전, id에서 엣지로 사전, 각 엣지의 빈도
    # output : 그래프
    def GenerateGraph(self, opt, dict_idx2keyword, dict_idx2edge, counter_edgeid2freq):
        # Vertex
        vertices = [i for i in range(len(dict_idx2keyword))]
        edges = dict_idx2edge

        self.obj_graph = Graph(vertex_attrs={"label":vertices}, edges=edges, directed=False)    
        for i, _ in enumerate(edges):
            self.obj_graph.es[i]["weight"] = counter_edgeid2freq[i]
        
        return self.obj_graph
    
    # input : 옵션, 그래프
    # output : 커뮤니티, 정제된 커뮤니티
    def FindCommunity(self, opt):
        self.community = self.obj_graph.community_multilevel(weights='weight', return_levels=False)
        self.org_subtopic_size = len(self.community)
                
        self.list_community = []
        for i, list_keyword in enumerate(self.community):
            if len(list_keyword) < opt.min_keyword: continue
            self.list_community.append( (list_keyword, str(i)) )
        self.cut_subtopic_size = len(self.list_community)
            
        return self.list_community
    
    def SetSubgraphdata(self, keywordidx2freq, edge2idx, edgeidx2freq):
                
        for list_keyword, _ in self.list_community:
            # Get edge_list from subgraph
            obj_subgraph = self.obj_graph.subgraph(list_keyword)            
            dict_newidx2keyidx = {}
            temp_keywordidx2freq = {}            
            for i, keyidx in enumerate(obj_subgraph.vs["label"]):
                dict_newidx2keyidx[i] = keyidx
                temp_keywordidx2freq[keyidx] = keywordidx2freq[keyidx]
            self.list_keyword2freq_insubtopics.append(temp_keywordidx2freq)
            
            temp_edgeidx2freq = {}
            list_edgeNweight = []
            for pair_edgeidx in obj_subgraph.get_edgelist():
                v1 = dict_newidx2keyidx[pair_edgeidx[0]]
                v2 = dict_newidx2keyidx[pair_edgeidx[1]]
                v1, v2 = sorted( [v1, v2] )
                tuple_edge = (v1, v2)
                if tuple_edge in edge2idx:
                    edge_idx = edge2idx[tuple_edge]
                    temp_edgeidx2freq[edge_idx] = edgeidx2freq[edge_idx]
            self.list_edge2freq_insubtopics.append(temp_edgeidx2freq)
