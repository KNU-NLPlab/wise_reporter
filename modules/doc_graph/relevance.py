import numpy as np
from operator import itemgetter


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
            relevance_value += idf_value * keyword2weight_indocs[key] / sum(
                keyword2weight_indocs)  # * keyword2weight_insubtopics[key]

        return relevance_value

    # input : document의 키워드 빈도, subtopic의 키워드 빈도, 키워드의 DF, 문서사이즈,
    #         문서의 키워드의 평균사이즈, 파라미터 k1, 파라미터 b
    # output : Relevance Score
    def Relevance_BM25(self, keyword2weight_indoc, keyword2weight_insubtopic, keyword2df, doc_size, avg_keywords_len,
                       k1=1.2, b=0.75):

        relevance_value = 0
        for key in keyword2weight_insubtopic.keys():
            idf_value = np.log((doc_size - keyword2df[key] + 0.5) / (keyword2df[key] + 0.5))
            relevance_value += idf_value * keyword2weight_indoc[key] * (k1 + 1) / \
                               (keyword2weight_indoc[key] + k1 * (
                                       1 - b + b * sum(keyword2weight_indoc.values()) / avg_keywords_len))

        return relevance_value

    def Relevance_BM25_Graph(self, keyword2weight_indoc, keyword2weight_insubtopic, keyword2df,
                             edge2weight_indoc, edge2weight_insubtopic, doc_size, avg_keywords_len, idx2edge, k1=1.2,
                             b=0.75):

        dict_key2BM25 = dict()
        relevance_value = 0
        for key in keyword2weight_insubtopic.keys():
            idf_value = np.log((doc_size - keyword2df[key] + 0.5) / (keyword2df[key] + 0.5))
            dict_key2BM25[key] = idf_value * keyword2weight_indoc[key] * (k1 + 1) / \
                                 (keyword2weight_indoc[key] + k1 * (
                                         1 - b + b * sum(keyword2weight_indoc.values()) / avg_keywords_len))

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

                graph_value = (temp_value / sum_edge_indoc + 1) / (
                        keyword2weight_indoc[v1] / sum_keyword_indoc * keyword2weight_indoc[
                    v2] / sum_keyword_indoc + 1)
                temp_relevance_value += (dict_key2BM25[v1] + dict_key2BM25[v2]) * graph_value
        relevance_value += temp_relevance_value

        return relevance_value

    ## modified by sspark ##
    # input : document의 키워드 빈도, subtopic의 키워드 빈도, 키워드의 DF, 문서사이즈,
    #         문서의 키워드의 평균사이즈, 파라미터 k1, 파라미터 b
    # output : Relevance Matrix (doc_size, subtopic_size)
    def CalculateRelevance(self, list_keyword2weight_indocs, dict_subgraph, cutting_level, select_rel=0):
        # LEAF node 서브 그래프 추출하기
        list_subgkey = [subgkey for subgkey in dict_subgraph.keys() if len(subgkey.split('/')) == cutting_level]
        list_subgraph = [dict_subgraph[subgkey]['graph'] for subgkey in list_subgkey]

        # 매트릭스 생성하기
        doc_size = len(list_keyword2weight_indocs)
        subtopic_size = len(list_subgraph)
        np_docsNsubtopic_rel = np.zeros((doc_size, subtopic_size))

        if select_rel == 0:
            list_graph_rel = [e.betweenness() for e in list_subgraph]
        elif select_rel == 1:
            list_graph_rel = [e.pagerank() for e in list_subgraph]
        elif select_rel == 2:
            list_graph_rel = [e.closeness() for e in list_subgraph]
        list_subgraphidx2idx = [{keyidx: i for i, keyidx in enumerate(sub_g.vs['label'])} for sub_g in list_subgraph]

        for i, keyword2weight_indoc in enumerate(list_keyword2weight_indocs):
            for j, a_subgraph in enumerate(list_subgraph):
                if select_rel == 0:
                    total_sum = 0
                    for keyidx in keyword2weight_indoc.keys():
                        if keyidx in list_subgraphidx2idx[j]:
                            total_sum += list_graph_rel[j][list_subgraphidx2idx[j][keyidx]] * keyword2weight_indoc[keyidx]
                    np_docsNsubtopic_rel[i][j] = total_sum
                elif select_rel == 1:
                    total_sum = 0
                    for keyidx in keyword2weight_indoc.keys():
                        if keyidx in list_subgraphidx2idx[j]:
                            total_sum += list_graph_rel[j][list_subgraphidx2idx[j][keyidx]] * keyword2weight_indoc[keyidx]
                    np_docsNsubtopic_rel[i][j] = total_sum
                elif select_rel == 2:
                    total_sum = 0
                    for keyidx in keyword2weight_indoc.keys():
                        if keyidx in list_subgraphidx2idx[j]:
                            total_sum += list_graph_rel[j][list_subgraphidx2idx[j][keyidx]] * keyword2weight_indoc[keyidx]
                    np_docsNsubtopic_rel[i][j] = total_sum

        self.np_docsNsubtopic_rel = np_docsNsubtopic_rel
        return np_docsNsubtopic_rel, list_subgkey

    ## modified by sspark ##

    def ExtractRepresentative(self, np_docsNsubtopic_rel, list_subgkey, dict_subgraph, doc_id_list, do_softmax=False):
        list_docidx_insubtopics = []

        doc_size = np_docsNsubtopic_rel.shape[0]
        community_size = np_docsNsubtopic_rel.shape[1]

        if do_softmax:
            def softmax(x):
                """Compute softmax values for each sets of scores in x."""
                e_x = np.exp(x - np.max(x, axis=1).reshape(x.shape[0], 1))
                return e_x / e_x.sum(axis=1).reshape(x.shape[0], 1)

            np_temp_docsNsubtopic_rel = softmax(np_docsNsubtopic_rel)
        else:            
            np_temp_docsNsubtopic_rel = np_docsNsubtopic_rel / np.sum(np_docsNsubtopic_rel, axis=1).reshape(doc_size, 1)

        # First, Sort relevance matrix with relevance score
        for i in range(community_size):
            idx_sorted_document = [i for i in range(doc_size)]
            idx_sorted_document = sorted(idx_sorted_document, key=lambda x: np_temp_docsNsubtopic_rel[x, i],
                                         reverse=True)

            subkey = list_subgkey[i]
            dict_subgraph[subkey]['documents'] = [doc_id_list[x] for x in idx_sorted_document]
            dict_subgraph[subkey]['docs_rel'] = [np_temp_docsNsubtopic_rel[x, i] for x in idx_sorted_document]
    
        return dict_subgraph
    ## modified by sspark ##
    
    ## modified by sspark ##
    def SetVariable(self, opt, dict_subgraph, list_subgkey, dict_newid2title):

        for subgkey in list_subgkey:
            a_graph = dict_subgraph[subgkey]['graph']
            list_score_keyword_pair = list(zip(a_graph.betweenness(), a_graph.vs['label']))
            sortedlist_score_keyword_pair = [word for score, word in
                                             sorted(list_score_keyword_pair, key=itemgetter(0), reverse=True)]
            dict_subgraph[subgkey]['keywords'] = sortedlist_score_keyword_pair[:opt.top_keyword_num]
            dict_subgraph[subgkey]['volume'] = len(a_graph.vs['label'])
            dict_subgraph[subgkey]['titles'] = [dict_newid2title[newsid] for newsid in dict_subgraph[subgkey]['documents'][:5]]
    ## modified by sspark ##