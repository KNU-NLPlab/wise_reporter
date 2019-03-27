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

        self.obj_graph = Graph(vertex_attrs={"label": vertices}, edges=edges, directed=False)
        for i, _ in enumerate(edges):
            self.obj_graph.es[i]["weight"] = counter_edgeid2freq[i]

        return self.obj_graph

    ## modified by sspark ##
    def RecursiveCommunity(self, obj_graph, cutting_level=3, list_cnt_level=[], min_keyword=0):
        community = obj_graph.community_multilevel(weights='weight', return_levels=False)

        dict_subgraph = dict()
        if len(list_cnt_level) != 0:
            dict_subgraph["/".join(list_cnt_level)] = {'graph': obj_graph}

        for i, a_subgraph in enumerate(community.subgraphs()):
            if len(a_subgraph.vs['label']) < min_keyword: continue

            if len(list_cnt_level) + 1 < cutting_level:
                dict_temp_subgraph = self.RecursiveCommunity(a_subgraph, cutting_level, list_cnt_level + [str(i)],
                                                             min_keyword)
                dict_subgraph.update(dict_temp_subgraph)
            else:
                dict_subgraph['/'.join(list_cnt_level + [str(i)])] = {'graph': a_subgraph}

        return dict_subgraph

    ## modified by sspark ##

    ## modified by sspark ##
    # input : 옵션, 그래프
    # output : 커뮤니티, 정제된 커뮤니티
    def FindCommunity(self, opt, cutting_level):
        dict_subgraph = self.RecursiveCommunity(obj_graph=self.obj_graph, cutting_level=cutting_level,
                                                list_cnt_level=[], min_keyword=opt.min_keyword)
        return dict_subgraph
    ## modified by sspark ##

#     def SetSubgraphdata(self, keywordidx2freq, edge2idx, edgeidx2freq):

#         for list_keyword, _ in self.list_community:
#             # Get edge_list from subgraph
#             obj_subgraph = self.obj_graph.subgraph(list_keyword)            
#             dict_newidx2keyidx = {}
#             temp_keywordidx2freq = {}            
#             for i, keyidx in enumerate(obj_subgraph.vs["label"]):
#                 dict_newidx2keyidx[i] = keyidx
#                 temp_keywordidx2freq[keyidx] = keywordidx2freq[keyidx]
#             self.list_keyword2freq_insubtopics.append(temp_keywordidx2freq)

#             temp_edgeidx2freq = {}
#             list_edgeNweight = []
#             for pair_edgeidx in obj_subgraph.get_edgelist():
#                 v1 = dict_newidx2keyidx[pair_edgeidx[0]]
#                 v2 = dict_newidx2keyidx[pair_edgeidx[1]]
#                 v1, v2 = sorted( [v1, v2] )
#                 tuple_edge = (v1, v2)
#                 if tuple_edge in edge2idx:
#                     edge_idx = edge2idx[tuple_edge]
#                     temp_edgeidx2freq[edge_idx] = edgeidx2freq[edge_idx]
#             self.list_edge2freq_insubtopics.append(temp_edgeidx2freq)
