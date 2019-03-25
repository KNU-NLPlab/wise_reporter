import os
import sys
import collections

class DocumentRaw:
    def __init__(self):
        self.info_in_listdoc = None  # 본문, 타이틀용, 나중에 지우면 됨
        self.keyword_in_listsent_in_listdoc = None  # 이거는 지우면 됨
        self.stopword = None  # 내가 만들면 됨
        
        self.list_keyword2freq_indocuments = []  # 문서 당 Noun Counter
        self.keyword2df = collections.Counter()
        self.keywordidx2frequency = collections.Counter()
        self.keyword2idx = {}
        self.idx2keyword = []
        
        self.list_edge2freq_indocuments = []  # 문서 당 Cooccur Counter     
        self.edgeidx2frequency = collections.Counter()
        self.edge2idx = {}
        self.idx2edge = []
        
        self.avg_keywords_len = 0  # 어절기준 문장의 average length
        
        self.doc_size = 0  # Doc 개수
        
    # input : 주제, 샘플사이즈
    # output : 문서가 가진 정보의 리스트
    def ProcessFilelist(self, opt):
        # get list of topic
        list_input_file = []
        if opt.input_topic == "all":
            list_input_file.extend(["bitcoin", "taxi", "thaad", "top", "u20"])
        else:
            list_input_file.extend([opt.input_topic])

        # get list of document [topic_num, doc_size, \info]
        list_document_obj = []
        for ele_input_topic in list_input_file:
            list_topic_document_obj = []
            input_path = opt.input_dir + "/{}.parse/".format(ele_input_topic)
            for json_file in sorted(os.listdir(input_path)):
                with open(os.path.join(input_path, json_file), "r") as fr:
                    json_obj = json.load(fr)
                    for doc_ele in json_obj:
                        list_topic_document_obj.append( (doc_ele['etri_contents'], doc_ele["title"], doc_ele["contents"]) )
            list_document_obj.append(list_topic_document_obj)

        # Shuffle Data list or sampling data [doc_size, \info]
        if opt.input_topic == "all" and opt.sample_size != 0:
            split_size = int(opt.sample_size/len(list_input_file))
            split_merge_list = []
            if opt.sample_size != 0:
                for list_topic_doc in list_document_obj:
                    random.shuffle(list_topic_doc)
                    split_merge_list.extend(list_topic_doc[:split_size])
            list_document_obj = split_merge_list
        elif opt.input_topic == "all" and opt.sample_size == 0:
            temp_list_document_obj = []
            for e in list_document_obj:
                temp_list_document_obj.extend(e)
            list_document_obj = temp_list_document_obj
        else:
            list_document_obj = list_document_obj[0]
            if opt.sample_size != 0:
                list_document_obj = list_document_obj[:opt.sample_size]
        random.shuffle(list_document_obj)
        self.info_in_listdoc = list_document_obj
        
    def SetStopword(self, stopword_path):
        # load stopwords
        # stopword_path = 'data/stopword.txt'
        with open(stopword_path, 'r', encoding='utf8') as fp:
            stopword_list = [l.strip().split('/') for l in fp.readlines()]

        cleaned_stopword = []
        for stopwords in stopword_list:
            if stopwords[0] == '@@':
                continue
            cleaned_stopword += stopwords
            cleaned_stopword = sorted(cleaned_stopword, key=lambda x: len(x.split(' ')))
        self.stopword = cleaned_stopword
        
    def SetMorphCorpus(self, opt):
        # cut and make keyword dictionary
        word_checker = re.compile(r'[^ ㄱ-ㅣ가-힣|a-zA-Z_|0-9]+')
        morph_corpus = []
        for doc_ele in self.info_in_listdoc:
            etri_contents_json = doc_ele[0]
            etri_contents = json.loads(etri_contents_json)

            moprh_doc = []
            for e in etri_contents['sentence'][:-1]:
                morph_sentence = []
                for morp_e in e['morp']:
                    if morp_e["type"].startswith("NNG") == True or morp_e["type"].startswith("NNP") == True:
                        if opt.word_check:
                            result = word_checker.findall(morp_e["lemma"])
                            if len(result) > 0: continue
                        morph_sentence.append(morp_e["lemma"])
                        
                # cut stopwords
                morph_sentence = ' '.join(morph_sentence)
                for stopword in self.stopword:
                    morph_sentence = morph_sentence.replace(stopword, '')
                morph_sentence = morph_sentence.strip().split(' ')
                moprh_doc += [morph for morph in morph_sentence if opt.termcnt_cut and len(morph) > 1]

            morph_corpus.append(moprh_doc)
        self.doc_size = len(self.info_in_listdoc)
        self.morph_corpus = morph_corpus

#         dictKeyword = corpora.Dictionary(morph_corpus)
#         print("Original vocab size : ", len(dictKeyword))

    # input : 한 문서의 에트리 분석 내용, 엣지의 카운트 개수
    # output : edge2id, idx2edge
    def GetListsentenceIndoc(self, opt, etri_contents, edge_cnt):
        
        # Edge tuple list                
        word_checker = re.compile(r'[^ ㄱ-ㅣ가-힣|a-zA-Z_|0-9]+')   
        
        list_sentence = []    
        set_morp = set()        
        keyword2freq_adoc = collections.Counter()
        edge2freq_adoc = collections.Counter()
        for e in etri_contents['sentence'][:-1]:
            list_morp = []
            list_edge = []
            for morp_e in e['morp']:
                if morp_e["type"].startswith("NNG") == True or morp_e["type"].startswith("NNP") == True:         
                    if opt.word_check:                                    
                        result = word_checker.findall(morp_e["lemma"])
                        if len(result) > 0: continue                            
                    if morp_e["lemma"] in self.keyword2idx:
                        morp_idx = self.keyword2idx[morp_e["lemma"]]
                        list_morp.append(morp_idx)

            if len(list_morp) == 0: continue            
            elif len(list_morp) > 1:
                len_morp = len(list_morp)
                for i in range(len_morp):                
                    for j in range(i,len_morp):
                        if list_morp[i] == list_morp[j]:
                            continue
                        sorted_morp = sorted([list_morp[i], list_morp[j]])
                        edge_tuple = (sorted_morp[0], sorted_morp[1])
                        if not edge_tuple in self.edge2idx:
                            self.idx2edge.append(edge_tuple)
                            self.edge2idx[edge_tuple] = edge_cnt
                            edge_cnt += 1
                        edge_idx = self.edge2idx[edge_tuple]
                        list_edge.append(edge_idx)

            len_morp = len(list_morp)
            for morp_idx in set(list_morp):
                value_count = list_morp.count(morp_idx)
                self.keywordidx2frequency[morp_idx] += value_count
                set_morp.add(morp_idx)
                self.avg_keywords_len += value_count
                
                keyword2freq_adoc[morp_idx] += value_count

            len_edge = len(list_edge)
            for edge_idx in set(list_edge):
                value_count = list_edge.count(edge_idx)
                self.edgeidx2frequency[edge_idx] += value_count
                
                edge2freq_adoc[edge_idx] += value_count

            list_sentence.append(list_morp)
        
        for morp_idx in set_morp:
            self.keyword2df[morp_idx] += 1        
        
        return list_sentence, edge_cnt, keyword2freq_adoc, edge2freq_adoc

    # input : 형태소분석의 코퍼스, 문서가 가진 정보 리스트
    # output : keyword2id, idx2keyword, 문서마다 키워드의 리스트
    def SetCorpusinfo(self, opt):
        # Cutting according to DF
        dictKeyword = corpora.Dictionary(self.morph_corpus)
        print("Original vocab size : ", len(dictKeyword))

        if opt.filter_df:
            dictKeyword.filter_extremes(int(opt.keyword_lower_bound), opt.keyword_upper_bound)        
        if opt.filter_df_rate:
            dictKeyword.filter_extremes(int(opt.keyword_lower_bound*len(self.morph_corpus)), opt.keyword_upper_bound)
        self.keyword2idx = dictKeyword.token2id
        self.idx2keyword = [k for k in self.keyword2idx.keys()]
        print("Trimmed vocab size  : ", len(dictKeyword))

        # Get keyword list in document list
        edge_cnt = 0
        keyword_in_listsent_in_listdoc = []
        for doc_ele in self.info_in_listdoc:
            etri_contients_json = json.loads(doc_ele[0])
            keyword_in_listsent, edge_cnt, keyword2freq_adoc, edge2freq_adoc = self.GetListsentenceIndoc(opt, etri_contients_json, edge_cnt)
            
            keyword_in_listsent_in_listdoc.append(keyword_in_listsent)
            self.list_keyword2freq_indocuments.append(keyword2freq_adoc)
            self.list_edge2freq_indocuments.append(edge2freq_adoc)
#             if opt.analysis:
#                 list_statistics_doc.append( {"node":len(counter_keyword),"edge_sum":sum(counter_edge.values()), "edge":len(counter_edge)} )
        self.keyword_in_listsent_in_listdoc = keyword_in_listsent_in_listdoc
    
        self.avg_keywords_len = self.avg_keywords_len / self.doc_size
