import os
import re
import sys
from collections import Counter
from operator import itemgetter

from modules.base_module import BaseModule

from modules.doc_graph.document_raw import DocumentRaw
from modules.doc_graph.document_graph import DocumentGraph
from modules.doc_graph.relevance import Relevance
from modules.doc_graph.analysis import Analysis
from modules.doc_graph.evaluation import Evaluation
from modules.doc_graph.utils import *


class DocGraphAnalyzer(BaseModule):
    '''
    Analyzer class to detect keyword set of communities in the graph of documents
    Args:
        topic (string): a keyword query
        df_bound (tuple of float): a lower and upper bound for keyword preprocessing
        min_keyword (int): a bound to use a community as a cluster
    '''
    def __init__(self, topic, df_bound=(0.02, 0.70), min_keyword=20):
        self.topic = topic
        self.df_bound = df_bound
        
        self.min_keyword = min_keyword
        
        self.raw_doc_obj = None
        self.docgraph_obj = None
        self.relevance_obj = None
        self.analsis_obj = None
        
        self.top_doc_id_list = None
        self.top_keyword_list = None
        
        self.viz_data = None

    def process_data(self, doc_info_list, top_doc_num=10, top_keyword_num=10):
        '''
        Overrided method from BaseModule class
        Extract top keyword and document id in several subtopics
        Args:
            doc_info_list (list): a list of document information dictionary
            top_doc_num (int): the number of documents to retrieve
            top_keyword_num (int): the number of keywords to retrieve
        '''
        doc_id_list = [doc_info['news_id'] for doc_info in doc_info_list]
        
        node_cnt_list, node2freq, node2df, node2idx = self._make_node_counter(doc_info_list)
        edge_cnt_list, edge2freq, edge2idx = self._make_cooccur_counter(doc_info_list, node2idx)
        
        raw_doc_obj = self._generate_doc_raw_obj(node_cnt_list, node2freq, node2df, node2idx, edge_cnt_list, edge2freq, edge2idx)
        
        sys.argv = []
        parser = argparse.ArgumentParser(sys.argv)
        opt = ParseOption(parser)
           
        opt.min_keyword = self.min_keyword
        opt.top_doc_num = top_doc_num
        opt.top_keyword_num = top_keyword_num
        
        docgraph_obj = DocumentGraph()
        _ = docgraph_obj.GenerateGraph(opt, raw_doc_obj.idx2keyword, raw_doc_obj.idx2edge, raw_doc_obj.edgeidx2frequency)
        _ = docgraph_obj.FindCommunity(opt)
        docgraph_obj.SetSubgraphdata(raw_doc_obj.keywordidx2frequency, raw_doc_obj.edge2idx, raw_doc_obj.edgeidx2frequency)
        
        rel_e = 1
        relevance_obj = Relevance()
        relevance_obj.CalculateRelevance(raw_doc_obj.list_keyword2freq_indocuments, docgraph_obj.list_keyword2freq_insubtopics,
                                     raw_doc_obj.keyword2df, raw_doc_obj.list_edge2freq_indocuments, docgraph_obj.list_edge2freq_insubtopics,
                                     raw_doc_obj.avg_keywords_len, raw_doc_obj.idx2edge, select_rel=rel_e)
        relevance_obj.ExtractRepresentative()

        analsis_obj = Analysis()
        analsis_obj.SetVariable(opt, relevance_obj.np_docsNsubtopic_rel, doc_id_list, docgraph_obj.community.subgraphs(),
                                docgraph_obj.list_community, raw_doc_obj.edgeidx2frequency, docgraph_obj.list_keyword2freq_insubtopics,
                                docgraph_obj.list_edge2freq_insubtopics, relevance_obj.docidx_insubtopics,
                                raw_doc_obj.idx2keyword, raw_doc_obj.idx2edge, 
                                docgraph_obj.org_subtopic_size, docgraph_obj.cut_subtopic_size)
        
        self.raw_doc_obj = raw_doc_obj
        self.docgraph_obj = docgraph_obj
        self.relevance_obj = relevance_obj
        self.analsis_obj = analsis_obj
        
        self.top_doc_id_list = [[doc_id_list[doc_idx] for doc_idx in top_doc_idx_list]
                                for top_doc_idx_list in analsis_obj.top_docidx_insubtopics]
        self.top_keyword_list = [[raw_doc_obj.idx2keyword[keyword_idx] for keyword_idx in top_keyword_list]
                                 for top_keyword_list in analsis_obj.top_keyword_insubtopics]
        
        print('Doc #  : {}'.format(len(doc_id_list)))
        print('Node # : {}'.format(len(raw_doc_obj.idx2keyword)))
        print('Edge # : {}\n'.format(len(raw_doc_obj.idx2edge)))
        
        zipped_data = zip(docgraph_obj.list_community, self.top_keyword_list)
        for (comm_word_list, comm_idx), top_keywords in zipped_data:
            print('Community {} : {:4} Keyword, ({})'.format(comm_idx, len(comm_word_list), '/'.join(top_keywords)))
        print()
            
    def get_viz_json(self, top_doc_info_list, node_cut=50, edge_cut=20):
        '''
        Overrided method from BaseModule class
        Generate json dicts for main page and detailed page and return them
        Args:
            top_doc_info_list (list): list of top document information dictionary
            node_cut (int): the maximum number of nodes to use for each cluster in visualization
            edge_cut (int): the maximum number of edges to use for each cluster in visualization
        Return:
            main_json
            detail_json
        '''        
        def verticesInSameCommunity(v1, v2):
            v1_comm = None
            v2_comm = None
            for idx_list, cluster_id in self.docgraph_obj.list_community:
                if v1 in idx_list:
                    v1_comm = cluster_id
                if v2 in idx_list:
                    v2_comm = cluster_id
                if v1_comm != v2_comm:
                    return False
                elif v1_comm == v2_comm and v1_comm != None:
                    return True
                
        assert self.raw_doc_obj is not None and self.docgraph_obj is not None and self.relevance_obj is not None, self.analsis_obj is not None
        
        node_idx_list = [sorted_keyword_list[:node_cut] for sorted_keyword_list in self.analsis_obj.sorted_keyword_insubtopics]
        # 살아남을 Edge의 node들 (Edge의 두 Node 모두 여기 있어야 Edge 살림)
        edge_node_idx_list = sum([sorted_keyword_list[:edge_cut] for sorted_keyword_list in self.analsis_obj.sorted_keyword_insubtopics], [])
        
        idx2word = self.raw_doc_obj.idx2keyword
        idx2edge = self.raw_doc_obj.idx2edge
        idx2edgefreq = self.raw_doc_obj.edgeidx2frequency
        
        cluster_info = {}
        node_info = []
        edge_info = []
        real_node_list = []
        
        # cluster, node 정보 추가
        zipped_info = zip(node_idx_list, self.top_keyword_list, top_doc_info_list)
        for i, (node_idxs, top_keywords, top_doc_infos) in enumerate(zipped_info):
            cluster_info[i] = {'node':node_idxs,
                               'top_keyword':top_keywords,
                               'top_doc_id':[top_doc_info['news_id'] for top_doc_info in top_doc_infos],
                               'top_title':[top_doc_info['newsTitle'] for top_doc_info in top_doc_infos],
                               'top_content':[top_doc_info['extContent'] for top_doc_info in top_doc_infos]}
            
            for node_idx in node_idxs:
                node_info.append([node_idx, idx2word[node_idx], i])
                real_node_list.append(node_idx)
        
        # edge 정보 추가
        for idx, edge_weight in idx2edgefreq.items():
            v1, v2 = idx2edge[idx]
            if v1 in real_node_list and v2 in real_node_list:
                if verticesInSameCommunity(v1, v2) or (v1 in edge_node_idx_list and v2 in edge_node_idx_list):
                    edge_info.append((v1, v2, edge_weight))
                    
        json_data = {'clusters':cluster_info,
                     'nodes':node_info,
                     'edges':edge_info}

        print('Viz Node # : ', len(node_info))
        print('Viz Edge # : ', len(edge_info))
        
        main_json = [min([doc_info['newsTitle'][:re.search('$|[\.…]', doc_info['newsTitle']).start()].strip()
                          for doc_info in doc_info_list], key=len) for doc_info_list in top_doc_info_list]
        detail_json = json_data
        
        return (main_json, detail_json)
        
    def make_summary(self, file_name='summary.txt', title=True, content=False):
        with open(os.path.join(self.output_path, file_name), 'w', encoding='utf8') as fp:
            for i, cluster_top_info in self.detailed_json['clusters'].items():
                fp.write('Community {} : {}\n'.format(i, ' / '.join(cluster_top_info['top_keyword'])))
                if title:
                    fp.write('\n'.join(cluster_top_info['top_title']))
                if content:
                    fp.write('\n'.join(cluster_top_info['top_content']))
                fp.write('\n\n')
    
    def _make_node_counter(self, doc_info_list):
    
        def node2counter(str_list):
            node_cnt = Counter()
            for s in str_list:
                word, cnt = s.split('=')
                # 11/13 added: remove 1 character word
                if len(word) != 1:
                    node_cnt[word] = int(cnt)
            return node_cnt
        
        node_cnt_list = [node2counter(doc_info['node']) for doc_info in doc_info_list]
        
        joined_noun_df_cnt = self._join_counter_list(node_cnt_list, key=lambda x: x.keys())
        joined_noun_cnt = self._join_counter_list(node_cnt_list)
        
        doc_size = len(doc_info_list)
        joined_noun_cnt, joined_noun_df_cnt = self._cut_vocab(joined_noun_cnt, joined_noun_df_cnt, doc_size, *self.df_bound)
        
        sorted_noun = sorted(joined_noun_cnt.items(), key=itemgetter(1), reverse=True)
        word2idx = {noun_info[0]:i for i, noun_info in enumerate(sorted_noun)}
        
        node_cnt_list = [Counter({word2idx[word]:cnt for word, cnt in node_cnt.items() if word in word2idx}) for node_cnt in node_cnt_list]
        noun_idx_df_cnt = {word2idx[noun]:df for noun, df in joined_noun_df_cnt.items()}
        noun_idx_cnt = {word2idx[noun]:tf for noun, tf in joined_noun_cnt.items()}
        
        return node_cnt_list, noun_idx_cnt, noun_idx_df_cnt, word2idx
    
    def _make_cooccur_counter(self, doc_info_list, word2idx):
        
        def edge2counter(str_list):
            edge_cnt = Counter()
            for s in str_list:
                word_pair, cnt = s.split('=')
                w1, w2 = word_pair[1:-1].split(',')
                # Cut된 단어가 하나라도 있으면 제외
                if w1 not in word2idx or w2 not in word2idx:
                    continue
                edge_cnt[tuple(sorted((word2idx[w1], word2idx[w2])))] = int(cnt)
            return edge_cnt
    
        edge_cnt_list = [edge2counter(doc['edge']) for doc in doc_info_list]
        joined_cooccur_cnt = self._join_counter_list(edge_cnt_list)
        
        sorted_cooccur = sorted(joined_cooccur_cnt.items(), key=itemgetter(1), reverse=True)
        edge2idx = {edge_info[0]:i for i, edge_info in enumerate(sorted_cooccur)}
        
        edge_cnt_list = [{edge2idx[edge]:cnt for edge, cnt in edge_cnt.items()} for edge_cnt in edge_cnt_list]
        edge_idx_cnt = {edge2idx[edge]:cooccur for edge, cooccur in joined_cooccur_cnt.items()}
        
        return edge_cnt_list, edge_idx_cnt, edge2idx
        
    def _load_stopword(self, path='stopword.txt'):
        with open(path, 'r', encoding='utf8') as fp:
            stopword_list = [l.strip().split('/') for l in fp.readlines()]
    
        cleaned_stopword = ['']
        for stopwords in stopword_list:
            if stopwords[0] == '@@':
                continue
            cleaned_stopword.append(stopwords[0])
        return cleaned_stopword

    def _cut_vocab(self, noun_cnt, noun_df_cnt, doc_size, lower_bound_rate, upper_bound_rate):
        lower_bound = int(doc_size * lower_bound_rate)
        upper_bound = int(doc_size * upper_bound_rate)
        
        cut_word_list = [word for word, df in noun_df_cnt.items() if df < lower_bound or df > upper_bound]
        
        for word in cut_word_list:
            noun_df_cnt.pop(word, None)
            noun_cnt.pop(word, None)
            
        return noun_cnt, noun_df_cnt
    
    def _join_counter_list(self, counter_list, key=lambda x: x):
        joined_counter = Counter()
        for cnt in counter_list:
            joined_counter.update(key(cnt))
        return joined_counter
    
    def _generate_doc_raw_obj(self, node_cnt_list, node2freq, node2df, node2idx,
                              edge_cnt_list, edge2freq, edge2idx):
        idx2node = [word for word in node2idx.keys()]
        idx2edge = [edge for edge in edge2idx.keys()]

        doc_size = len(node_cnt_list)
        avg_morph_len = 0
        
        doc_obj = DocumentRaw()
        
        doc_obj.info_in_listdoc = None
        doc_obj.keyword_in_listsent_in_listdoc = None
        doc_obj.stopword = None
        
        doc_obj.list_keyword2freq_indocuments = node_cnt_list
        doc_obj.keyword2df = node2df
        doc_obj.keywordidx2frequency = node2freq
        doc_obj.keyword2idx = node2idx
        doc_obj.idx2keyword = idx2node

        doc_obj.list_edge2freq_indocuments = edge_cnt_list
        doc_obj.edgeidx2frequency = edge2freq
        doc_obj.edge2idx = edge2idx
        doc_obj.idx2edge = idx2edge

        doc_obj.avg_keywords_len = avg_morph_len
        doc_obj.doc_size = doc_size
        
        return doc_obj