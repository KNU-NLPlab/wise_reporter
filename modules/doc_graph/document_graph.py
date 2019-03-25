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
    
    ## modified by sspark ##
    def RecursiveCommunity(self, obj_graph, goal_level=3, list_cnt_level=[], min_keyword=0):
        community = obj_graph.community_multilevel(weights='weight', return_levels=False)
        
        list_community = []
        list_subgraph = []
        list_org_subtopic_size = []        
        if not len(list_cnt_level) + 1 < goal_level:
            list_org_subtopic_size.append( len(community) )
            
        for i, (a_subgraph, list_keyword) in enumerate(zip(community.subgraphs(), community)):
            if len(list_keyword) < min_keyword: continue
            
            if len(list_cnt_level) + 1 < goal_level:                
                list_temp_community, list_temp_subgraph, list_temp_org_subtopic_size = self.RecursiveCommunity(a_subgraph,
                                                                                                              goal_level,
                                                                                                              list_cnt_level + [str(i)],
                                                                                                              min_keyword)
                list_community.extend(list_temp_community)
                list_subgraph.extend(list_temp_subgraph)
                list_org_subtopic_size.extend(list_temp_org_subtopic_size)
            else:
                self.org_subtopic_size = len(community)
                list_community.append( (list_keyword, "/".join(list_cnt_level + [str(i)]) ) )
                list_subgraph.append( a_subgraph )
                
        
        return list_community, list_subgraph, list_org_subtopic_size
    ## modified by sspark ##
    
    ## modified by sspark ##
    # input : 옵션, 그래프
    # output : 커뮤니티, 정제된 커뮤니티
    def FindCommunity(self, opt, hierarchy, goal_level):
        
        if hierarchy == 0:
            self.community = self.obj_graph.community_multilevel(weights='weight', return_levels=False)
            self.org_subtopic_size = len(self.community)
            
            self.list_community = []
            for i, list_keyword in enumerate(self.community):
                if len(list_keyword) < opt.min_keyword: continue
                self.list_community.append( (list_keyword, str(i)) )
            self.cut_subtopic_size = len(self.list_community)            
            self.list_subgraph = self.community.subgraphs()
            
        elif hierarchy == 1:
            self.community = self.obj_graph.community_multilevel(weights='weight', return_levels=True)
            
            self.list_community = []
            for i, list_cluster in enumerate(self.community):
                for j, list_keyword in enumerate(list_cluster):
                    self.org_subtopic_size += 1
                    if len(list_keyword) < opt.min_keyword: continue
                    self.list_community.append( (list_keyword, "{}/{}".format(i,j)) )
            self.cut_subtopic_size = len(self.list_community)
            
            self.list_subgraph = []
            for comm in self.community:
                for comm_subgraph in comm.subgraphs():
                    self.list_subgraph.append(comm_subgraph)
                    
        elif hierarchy == 2:            
            list_community, list_subgraph, list_org_subtopic_size = self.RecursiveCommunity(obj_graph=self.obj_graph,
                                                                                            goal_level=goal_level,
                                                                                            min_keyword=opt.min_keyword)
            self.list_community = list_community
            self.list_subgraph = list_subgraph
            self.org_subtopic_size = sum(list_org_subtopic_size)
            self.cut_subtopic_size = len(self.list_community)
            
        return self.list_community
    ## modified by sspark ##
    

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
