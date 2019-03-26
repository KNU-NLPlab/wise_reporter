import argparse
import time

def Main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    opt = ParseOption(parser)
    
    pkl_path = opt.output_path + "_fixed.pkl"
    
    if os.path.isfile(pkl_path) == True:
        print("pkl")
        if opt.input_topic == "top":
            list_dict_docid2rellabel = []
            rel_data_path = "data/raw/rel_label_top.txt"
            with open(rel_data_path, "r") as fr:
                for line in fr.readlines():
                    if line.startswith("## subtopic"):
                        subtopic_label = int(line[12])
                        list_dict_docid2rellabel.append(dict())
                        continue
                    else:
                        splited_line = line.split()
                        docid = splited_line[0]
                        rellabel = splited_line[1]
                        if docid in list_dict_docid2rellabel[subtopic_label]:
                            print("subtopic",subtopic_label)
                            print(docid, list_dict_docid2rellabel[subtopic_label][docid],rellabel)
                        list_dict_docid2rellabel[subtopic_label][int(docid)] = int(rellabel)

        with open(pkl_path, "rb") as fr:
            pkl_obj = pickle.load(fr)
            raw_doc_obj, docgraph_obj = pkl_obj
    else:
        
        stopword_path = 'data/stopword.txt'

        raw_doc_obj = DocumentRaw()
        raw_doc_obj.ProcessFilelist(opt)
        raw_doc_obj.SetStopword(stopword_path)
        raw_doc_obj.SetMorphCorpus(opt)
        raw_doc_obj.SetCorpusinfo(opt)

        docgraph_obj = DocumentGraph()
        _ = docgraph_obj.GenerateGraph(opt, raw_doc_obj.idx2keyword, raw_doc_obj.idx2edge, raw_doc_obj.edgeidx2frequency)
        _ = docgraph_obj.FindCommunity(opt)
        docgraph_obj.SetSubgraphdata(raw_doc_obj.keywordidx2frequency, raw_doc_obj.edge2idx, raw_doc_obj.edgeidx2frequency)
        
        with open(pkl_path, "wb") as fw:
            pkl_obj = (raw_doc_obj, docgraph_obj)
            pickle.dump(pkl_obj, fw)
    
    fw = open(opt.output_path + "_document.txt", "w")
    relevance_list = [0, 1, 2, 3]    
    for rel_e in relevance_list:
        relevance_obj = Relevance()
        relevance_obj.CalculateRelevance(raw_doc_obj.list_keyword2freq_indocuments, docgraph_obj.list_keyword2freq_insubtopics,
                                     raw_doc_obj.keyword2df, raw_doc_obj.list_edge2freq_indocuments, docgraph_obj.list_edge2freq_insubtopics,
                                     raw_doc_obj.avg_keywords_len, raw_doc_obj.idx2edge, select_rel=rel_e)
        relevance_obj.ExtractRepresentative()
        
        analsis_obj = Analysis()
        analsis_obj.SetVariable(opt, relevance_obj.np_docsNsubtopic_rel, raw_doc_obj.info_in_listdoc, raw_doc_obj.keywordidx2frequency,
                                raw_doc_obj.edgeidx2frequency,
                                docgraph_obj.list_keyword2freq_insubtopics, docgraph_obj.list_edge2freq_insubtopics, relevance_obj.docidx_insubtopics,
                                raw_doc_obj.idx2keyword, raw_doc_obj.idx2edge, 
                                docgraph_obj.org_subtopic_size, docgraph_obj.cut_subtopic_size)
        # analsis_obj.PrintFileWithTitle(opt, opt.output_path+"_{}.txt".format(rel_e), start_time)
        analsis_obj.PrintFileWithContents(opt, opt.output_path+"_content_{}.txt".format(rel_e), start_time)
        
        for i, docidx_insubtopic in enumerate(relevance_obj.docidx_insubtopics):
                        
            list_docid2relevance = [0 for e in docidx_insubtopic]
            for docid in docidx_insubtopic[:opt.represent_size]:
                # print(docid, list_dict_docid2rellabel[i].get(docid))
                if docid in list_dict_docid2rellabel[i]:
                    rel = list_dict_docid2rellabel[i][docid]
                else:
                    rel = 0
                list_docid2relevance[docid] = rel
            # print(list_docid2relevance)
            
            obj_eval = Evaluation(docidx_insubtopic, list_docid2relevance)
            print(i,"/", obj_eval.MAP(), obj_eval.NDCGp())
    fw.close()

def p1():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    opt = ParseOption(parser)
    
    stopword_path = 'data/stopword.txt'

    raw_doc_obj = DocumentRaw()
    raw_doc_obj.ProcessFilelist(opt)
    raw_doc_obj.SetStopword(stopword_path)
    raw_doc_obj.SetMorphCorpus(opt)
    raw_doc_obj.SetCorpusinfo(opt)

    docgraph_obj = DocumentGraph()
    _ = docgraph_obj.GenerateGraph(opt, raw_doc_obj.idx2keyword, raw_doc_obj.idx2edge, raw_doc_obj.edgeidx2frequency)
    _ = docgraph_obj.FindCommunity(opt)
    docgraph_obj.SetSubgraphdata(raw_doc_obj.keywordidx2frequency, raw_doc_obj.edge2idx, raw_doc_obj.edgeidx2frequency)        
    
    relevance_list = [0, 1, 2]
    for rel_e in relevance_list:
        relevance_obj = Relevance()
        relevance_obj.CalculateRelevance(raw_doc_obj.list_keyword2freq_indocuments, docgraph_obj.list_keyword2freq_insubtopics,
                                     raw_doc_obj.keyword2df, raw_doc_obj.list_edge2freq_indocuments, docgraph_obj.list_edge2freq_insubtopics,
                                     raw_doc_obj.avg_keywords_len, raw_doc_obj.idx2edge, select_rel=rel_e)
        relevance_obj.ExtractRepresentative()

        analsis_obj = Analysis()
        analsis_obj.SetVariable(opt, relevance_obj.np_docsNsubtopic_rel, raw_doc_obj.info_in_listdoc, raw_doc_obj.keywordidx2frequency,
                                raw_doc_obj.edgeidx2frequency,
                                docgraph_obj.list_keyword2freq_insubtopics, docgraph_obj.list_edge2freq_insubtopics, relevance_obj.docidx_insubtopic,
                                raw_doc_obj.idx2keyword, raw_doc_obj.idx2edge, 
                                docgraph_obj.org_subtopic_size, docgraph_obj.cut_subtopic_size)
        analsis_obj.PrintFileWithTitle(opt, opt.output_path+"_{}.txt".format(rel_e), start_time)
        
'''
    - parse argument which this program need to have
    input : argument parser object
    output : option of parsed argument
'''
def ParseOption(parser):
    group = parser.add_argument_group('preprocess')
    group.add_argument('-input_dir', type=str, default="data/raw", help='input_dir')
    group.add_argument('-input_topic', type=str, default="u20", help='bitcoin, taxi, thaad, top, u20, all')    
    group.add_argument('-output_path', type=str, default="data", help='output_path')
    group.add_argument('-sample_size', type=int, default=0, help='Sampling with size')
    group.add_argument('-cut_k', type=int, default=-1, help='Cut the number of keyword by co-occurence')
    group.add_argument('-filter_df', action="store_true", default=False, help='')
    group.add_argument('-filter_df_rate', action="store_true", default=False, help='')
    group.add_argument('-keyword_lower_bound', type=float, default=0, help='output_path')
    group.add_argument('-keyword_upper_bound', type=float, default=1, help='output_path')
    
    group = parser.add_argument_group('method')
    group.add_argument('-represent_size', type=int, default=5, help='size of represent document')    
    group.add_argument('-min_keyword', type=int, default=20, help='min size of keyword in cluster')
    group.add_argument('-view_keyword', type=int, default=20, help='want to view keyword in community')
    group.add_argument('-top_keyword_num', type=int, default=10, help='top keyword #')
    group.add_argument('-relevance_choice', type=int, default=0, help='choice relevance score method')
    
    group = parser.add_argument_group('additional')
    group.add_argument('-analysis', action="store_true", default=False, help='Visually Analyze of graph size accoding to increment of node size')
    group.add_argument('-termcnt_cut', action="store_true", default=True, help='')
    group.add_argument('-word_check', action="store_true", default=True, help='') 
    
    return parser.parse_args()