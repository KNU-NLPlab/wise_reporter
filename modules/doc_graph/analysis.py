from operator import itemgetter

class Analysis:
    def __init__(self):
        self.relevance_indocs = None
                
        self.top_keyword_insubtopics = None
        self.top_edge_insubtopics = None
        self.top_docidx_insubtopics = None
        
        self.vertex_num = None
        self.edge_num = None
        self.org_subtopic_num = None
        self.cut_subtopic_num = None
        
    def SetVariable(self, opt, np_docsNsubtopics_rel, info_in_listdoc, communities_subgraph,
                    list_community, edgeidx2freq, keyidx2freq_insubtopics,
                    edgeidx2freq_insubtopics, docidx_insubtopics, 
                    idx2keyword, idx2edge,
                    org_subtopic_num, cut_subtopic_num):
            
        self.relevance_indocs = np_docsNsubtopics_rel.tolist()        
        
        ## 서브토픽안의 Top Keyword
        self.sorted_keyword_insubtopics = []
        for idx in range(len(list_community)):
            # betweenness, closeness, eigenvector_centrality, 
            keyword_score_pair_list = list(zip(communities_subgraph[idx].betweenness(),
                                               [idx for idx in list_community[idx][0]]))
            self.sorted_keyword_insubtopics.append([word for score, word in sorted(keyword_score_pair_list, key=itemgetter(0), reverse=True)])
        self.top_keyword_insubtopics = [keyword_list[:opt.top_keyword_num] for keyword_list in self.sorted_keyword_insubtopics]

        ## 서브토픽안의 Top Edge
        top_edge_insubtopics = []
        for edgeidx2freq in edgeidx2freq_insubtopics: # 
            sorted_edgeidx_freq = sorted(edgeidx2freq.keys(), reverse=True)
            temp_edge_list = []
            for edgeidx in sorted_edgeidx_freq[:opt.view_keyword]:
                v1, v2 = idx2edge[edgeidx]                
                temp_edge_list.append( "({},{})".format(idx2keyword[v1], idx2keyword[v2]) )
            top_edge_insubtopics.append(temp_edge_list)      
               
        top_docidx_insubtopics = []
        for docidxs in docidx_insubtopics:
            top_docidx_insubtopics.append(docidxs[:opt.top_doc_num])       
            
        self.top_edge_insubtopics = top_edge_insubtopics
        self.top_docidx_insubtopics = top_docidx_insubtopics
        
        self.vertex_num = len(idx2keyword)
        self.edge_num = len(edgeidx2freq)
        
        self.org_subtopic_num = org_subtopic_num
        self.cut_subtopic_num = cut_subtopic_num
            