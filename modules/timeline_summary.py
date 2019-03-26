# -*- coding: utf-8 -*-
# peakutil, burst

import os, sys, json, shutil
import time

import collections
from collections import OrderedDict

from modules.timeline_summ import makeTimeline
from modules.timeline_summ import getKeyword
from modules.timeline_summ import makePreCandSent
from modules.timeline_summ import redundancyCheck
from modules.timeline_summ import makeSummary

from modules.timeline_sum.cleanDirectory import cleanDirectory


def summarization(doc_save_list, outPath, query, timeline_threshold=0.3, phrase_threshold=17) :
    start_time = time.time()
    dataCommonPath = outPath
    
    cleanDirectory(outPath)
    
    tempDic = outPath + '/temp/'

    # A. make Timeline
    cleanDirectory(tempDic)
    # print('Phase 1. Make Timeline')
    
    publicTimelineSet, burstTimeSet, timelineSet = makeTimeline.main(doc_save_list, timeline_threshold)

    # B. getKeyword
    tfScores, sentencesSets = getKeyword.main(timelineSet, doc_save_list, query)

    # C. makePrecandidateSentence
    makePreCandSent.main(tfScores, sentencesSets, tempDic, query, phrase_threshold)

    # C-sub. removeRebundancy
    redundancyCheck.main(tempDic)

    # D. makeSummary
    outPath = dataCommonPath + '/'

    summary, beam_search = makeSummary.main(tfScores, tempDic, outPath, timelineSet, burstTimeSet, query)
    
    cleanDirectory(tempDic)

    print(' >> response time : %.2f'%(time.time() - start_time))


    return summary, beam_search

def preprocess(doc_info_list, doc_slice=1000) :

    totalDic = 0
    doc_save_list = []

    for elemNews in doc_info_list :
        _id = elemNews['id']
        date = elemNews['writetime'] + '_'
        doc_save_list.append(elemNews)

        if totalDic == doc_slice :
            return doc_save_list
        
    return doc_save_list


if __name__ == '__main__' :

    # Essential Input #
    query = '탑 대마초' #
    inPath = '' # downloaded path
    outPath = '' # final output path

    # Optional Input #
    doc_slice = 200 # initial : 200
    timeline_threshold = 0.3 # initial : 0.3
    phrase_threshold = 20 # initial : 20

    prePath = outPath + '/refine_' + query.split()[0] + '/' # '/workspace/dataset/refine_' + query.split()[0] + '/d.etlink/f.final/'

    preprocess(inPath, prePath, doc_slice)

    summary, beam_search = summarization(prePath, outPath, query, doc_slice, timeline_threshold, phrase_threshold)