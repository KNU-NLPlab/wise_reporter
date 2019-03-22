class Relevance:
    def __init__(self):        
        self.np_docsNfeature = None
        self.docidx_insubtopics = None
        
    # input : document의 키워드 빈도, subtopic의 키워드 빈도
    # output : Relevance Score
    def Relevance_TF(self, keyword2weight_indocs, keyword2weight_insubtopics):
        
        relevance_value = 0
        for key in keyword2weight_insubtopics.keys():            
            relevance_value += keyword2weight_indocs[key] * keyword2weight_insubtopics[key]
        
        return relevance_value
    
    def Relevance_TFIDF(self, keyword2weight_indocs, keyword2weight_insubtopics, keyword2df, doc_size):
        
        relevance_value = 0
        for key in keyword2weight_insubtopics.keys():
            idf_value = (doc_size - keyword2df[key] + 0.5) / (keyword2df[key] + 0.5)
            relevance_value += idf_value * keyword2weight_indocs[key] * keyword2weight_insubtopics[key]
        
        return relevance_value
    
    # input : document의 키워드 빈도, subtopic의 키워드 빈도, 키워드의 DF, 문서사이즈,
    #         문서의 키워드의 평균사이즈, 파라미터 k1, 파라미터 b
    # output : Relevance Score
    def Relevance_BM25(self, keyword2weight_indoc, keyword2weight_insubtopic, keyword2df, doc_size, avg_keywords_len, k1=1.2, b=0.75):
        
        relevance_value = 0
        for key in keyword2weight_insubtopic.keys():
            idf_value = np.log( (doc_size - keyword2df[key] + 0.5) / (keyword2df[key] + 0.5) )
            relevance_value += idf_value * keyword2weight_indoc[key] * (k1 + 1) /\
            ( keyword2weight_indoc[key] + k1 * (1 - b + b * sum(keyword2weight_indoc.values()) / avg_keywords_len) )

        return relevance_value
    
    def Relevance_BM25_Graph(self, keyword2weight_indoc, keyword2weight_insubtopic, keyword2df, 
                             edge2weight_indoc, edge2weight_insubtopic, doc_size, avg_keywords_len, idx2edge, k1=1.2, b=0.75):
                
        dict_key2BM25 = dict()
        relevance_value = 0
        for key in keyword2weight_insubtopic.keys():
            idf_value = np.log( (doc_size - keyword2df[key] + 0.5) / (keyword2df[key] + 0.5) )                        
            dict_key2BM25[key] = idf_value * keyword2weight_indoc[key] * (k1 + 1) /\
            ( keyword2weight_indoc[key] + k1 * (1 - b + b * sum(keyword2weight_indoc.values()) / avg_keywords_len) )
            
            # relevance_value += idf_value * keyword2weight_indoc[key] * (k1 + 1) /\
            # ( keyword2weight_indoc[key] + k1 * (1 - b + b * sum(keyword2weight_indoc.values()) / avg_keywords_len) )
        
        temp_relevance_value = 0
        
        sum_edge_indoc = sum(edge2weight_indoc.values())
        sum_keyword_indoc = sum(keyword2weight_indoc.values())
        for edgeid in edge2weight_insubtopic.keys():
            if edgeid in edge2weight_indoc:
                # print(edge_tuple, idx2edge([edge_tuple])
                v1, v2 = idx2edge[edgeid]

                if edge2weight_indoc.get(edgeid) == None:
                    temp_value = 0
                else:
                    temp_value = edge2weight_indoc.get(edgeid)

                graph_value = (temp_value/sum_edge_indoc + 1) / ( keyword2weight_indoc[v1]/sum_keyword_indoc * keyword2weight_indoc[v2]/sum_keyword_indoc + 1)
                temp_relevance_value += (dict_key2BM25[v1] + dict_key2BM25[v2]) * graph_value
        relevance_value += temp_relevance_value

        return relevance_value
    
    # input : document의 키워드 빈도, subtopic의 키워드 빈도, 키워드의 DF, 문서사이즈,
    #         문서의 키워드의 평균사이즈, 파라미터 k1, 파라미터 b
    # output : Relevance Matrix (doc_size, subtopic_size)
    def CalculateRelevance(self, list_keyword2weight_indocs, list_keyword2weight_insubtopics, keyword2df,
                           list_edge2weight_indocs, list_edge2weight_insubtopics, avg_keywords_len, idx2edge, select_rel=0):
        doc_size = len(list_keyword2weight_indocs)
        subtopic_size = len(list_keyword2weight_insubtopics)
        
        np_docsNsubtopic_rel = np.zeros( (doc_size, subtopic_size) )
        for i, (keyword2weight_indoc, edge2weight_indoc) in enumerate(zip(list_keyword2weight_indocs, list_edge2weight_indocs)):
            for j, (keyword2weight_insubtopic, edge2weight_insubtopic) in enumerate(zip(list_keyword2weight_insubtopics, list_edge2weight_insubtopics)):
                if select_rel == 0:
                    np_docsNsubtopic_rel[i][j] = self.Relevance_TF(keyword2weight_indoc, keyword2weight_insubtopic)
                if select_rel == 1:
                    np_docsNsubtopic_rel[i][j] = self.Relevance_TFIDF(keyword2weight_indoc, keyword2weight_insubtopic, keyword2df, doc_size)
                elif select_rel == 2:
                    np_docsNsubtopic_rel[i][j] = self.Relevance_BM25(keyword2weight_indoc, keyword2weight_insubtopic, keyword2df, doc_size, avg_keywords_len)
                elif select_rel == 3:
                    np_docsNsubtopic_rel[i][j] = self.Relevance_BM25_Graph(keyword2weight_indoc, keyword2weight_insubtopic, keyword2df, 
                                     edge2weight_indoc, edge2weight_insubtopic, doc_size, avg_keywords_len, idx2edge)
        self.np_docsNsubtopic_rel = np_docsNsubtopic_rel
    
    def ExtractRepresentative(self):   
        list_docidx_insubtopics = []

        doc_size = self.np_docsNsubtopic_rel.shape[0]
        community_size = self.np_docsNsubtopic_rel.shape[1]

        temp_idx_community = []
        # First, Sort relevance matrix with relevance score
        for i in range(community_size):
            idx_sorted_document = [i for i in range(doc_size)]
            idx_sorted_document = sorted(idx_sorted_document, key=lambda x:self.np_docsNsubtopic_rel[x,i],reverse=True)
#             temp_idx_community.append(idx_sorted_document)
            list_docidx_insubtopics.append(idx_sorted_document)

#         # Second, Normalize sorted relevance score of each community in a document
#         temp_value = np.sum(self.np_docsNsubtopic_rel, axis=1).reshape(doc_size, 1)
#         np_norm_docsNsubtopic_rel = self.np_docsNsubtopic_rel / temp_value

#         # Third, Sort processed relevance matrix with normalized and sorted relevance score
#         for i in range(community_size):
#             idx_sorted_document = temp_idx_community[i]
#             idx_sorted_document = sorted(idx_sorted_document, key=lambda x:np_norm_docsNsubtopic_rel[x,i],reverse=True)
#             list_docidx_insubtopics.append(idx_sorted_document)

        self.docidx_insubtopics = list_docidx_insubtopics
