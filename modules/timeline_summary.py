# -*- coding: utf-8 -*-
# peakutil, burst

import os, sys, json, shutil
import time

import collections
from collections import OrderedDict

from modules.base_module import BaseModule

from modules.timeline_summ import makeTimeline
from modules.timeline_summ import getKeyword
from modules.timeline_summ import makePreCandSent
from modules.timeline_summ import redundancyCheck
from modules.timeline_summ import makeSummary
from modules.timeline_summ import bertCompression

from modules.timeline_summ.cleanDirectory import cleanDirectory


class TimelineSummary(BaseModule):
    '''
    Summarizer class from the aspect of time
    Args:
        topic (string): a keyword query
        out_path (string): a output path to save json files
        doc_slice (int): a number of document size to be trimmed
    '''
    def __init__(self, topic, out_path, doc_slice=1000):
        super(TimelineSummary, self).__init__(topic, out_path)
        self.doc_slice = 1000
        

    def process_data(self, topic, doc_info_list, timeline_threshold=0.5, phrase_threshold=20):
        '''
        Overrided method from BaseModule class
        Extract top keyword and document id in several subtopics
        Args:
            top_doc_morph_sentence_list (list): a list of morph sentences in top documents
            top_keyword_list (list): a list of keywords of top documents
        '''
        self.topic = topic
        doc_save_list = self._preprocess(doc_info_list, len(doc_info_list))
        
        # summary -> demo main page
        # beam_search -> detail page
        self.main_json, self.detail_json = self._summarization(doc_save_list, timeline_threshold, phrase_threshold)
        return self.main_json, self.detail_json
    
    def generate_json(self):
        '''
        Overrided method from BaseModule class
        Return two json dictionary for visualization
        Args:
        '''
        return self.main_json, self.detail_json

    def _preprocess(self, doc_info_list, doc_slice=1000):
        totalDic = 0
        doc_save_list = []

        for elemNews in doc_info_list:
            #elemNews["id"] = elemNews.pop("news_id")
            elemNews["writetime"] = elemNews.pop("postingDate")
            doc_save_list.append(elemNews)

            if totalDic == doc_slice :
                return doc_save_list

        return doc_save_list

    def get_parameters(self, topic):
        if topic == '트럼프 유가 OPEC':
            timeline_threshold = 0.4
            phrase_threshold = 20
            onlyburst = False
            alpha = 3
            beta = 0
        elif topic == '우리나라 외환보유액 감소':
            timeline_threshold = 0.5
            phrase_threshold = 20
            onlyburst = True
            alpha = 1 / 3
            beta = 0
        elif topic == '브렉시트 메이 총리 英':
            timeline_threshold = 0.5
            phrase_threshold = 20
            onlyburst = True
            alpha = 1 / 3
            beta = 0
        elif topic == '미국 중국 무역':
            timeline_threshold = 0.5
            phrase_threshold = 20
            onlyburst = True
            alpha = 1 / 3
            beta = 0
        else:
            timeline_threshold = 0.4
            phrase_threshold = 20
            onlyburst = False
            alpha = 1 / 3
            beta = 0
#             timeline_threshold = 0.5
#             phrase_threshold = 20
#             onlyburst = True
#             alpha = 1 / 3
#             beta = 0
        return timeline_threshold, phrase_threshold, onlyburst, alpha, beta

    def _summarization(self, doc_save_list, timeline_threshold, phrase_threshold):
        '''
        Args:
            timeline_threshold : Burst의 임계치. 높이면 속도 증가
            phrase_threshold : 문장 길이 임계치. 줄이면 속도 증가
        '''
        timeline_threshold, phrase_threshold, onlyburst, alpha, beta = self.get_parameters(self.topic)


        start_time = time.time()
        
        timeline_out_path = os.path.join(self.out_path, "timeline/")

        cleanDirectory(timeline_out_path)

        # A. make Timeline
        tempDic = timeline_out_path + 'temp/'
        cleanDirectory(tempDic)
        
        publicTimelineSet, burstTimeSet, timelineSet = makeTimeline.main(doc_save_list, timeline_threshold, alpha = alpha, beta = beta)
        print("A : ", time.time()-start_time)
        start_time = time.time()

        # B. getKeyword
        tfScores, sentencesSets = getKeyword.main(timelineSet, doc_save_list, self.topic, onlyburst = onlyburst)
        print("B : ", time.time()-start_time)
        start_time = time.time()

        # C. makePrecandidateSentence
        makePreCandSent.main(tfScores, sentencesSets, tempDic, self.topic, phrase_threshold)
        print("C : ", time.time()-start_time)
        start_time = time.time()

        # C-sub. removeRebundancy
        redundancyCheck.main(tempDic)
        print("C-sub : ", time.time()-start_time)
        start_time = time.time()

        # D. makeSummary
        summary, beam_search = makeSummary.main(tfScores, tempDic, timeline_out_path, timelineSet, burstTimeSet, self.topic)
        print("D : ", time.time()-start_time)

        cleanDirectory(tempDic)

        # E. BERT compression
        summary = bertCompression.main(summary)
        print("E : ", time.time()-start_time)


        return summary, beam_search
