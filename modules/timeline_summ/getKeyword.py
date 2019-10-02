# -*- coding: utf-8 -*-
import os, sys, json
import collections
from collections import OrderedDict

def loadStopword(stopwordPath = 'stopword.txt') :
    stopwordPath = os.path.dirname(os.path.realpath(__file__)) + '/stopword.txt'
    stopword = []
    with open(stopwordPath, 'r', encoding = 'utf-8') as f:
        for line in f :
            stopword.append(line.rstrip())

    return stopword

def collectSents(timelineSet) :
    sentenceSets  = {}
    for time in timelineSet :
        sentenceSets[time] = []

    return sentenceSets

def isEnglish(input_s) :
    convert_s = ''
    for input_c in input_s :
        if ord('a') <= ord(input_c.lower()) <= ord('z') :
            convert_s += input_c.lower()
        else :
            convert_s += input_c

    return convert_s

def main(timelineSet, doc_save_list, query) :

    stopword = loadStopword()
    
    # A. Collect Sentences

    queryPhrase = query
    queryPiece = queryPhrase.split()
    
    sentenceSets = collectSents(timelineSet)
    
    for tmpdic in doc_save_list:
        sentences = tmpdic['extract']
        for sentence in sentences :
            if 'time' not in sentence or 'event' not in sentence :
                continue
            if len(sentence['time']) == 0 :
                continue

            words, dps, morph =  [], [], []
            for wordidx,  word in enumerate(sentence['word']) :
                morphs = word['morph_ev'].split('+')
                for _morph in morphs:
                    _morph = _morph.split('/')
                    wordtxt = _morph[0]
                    _morph = _morph[-1]

                    if _morph == '' :
                        continue

                    if _morph[0] == 'N' and _morph != 'NR' :
                        if wordtxt not in stopword and 'timex' not in word:
                            if len(words) != 0 and len(queryPiece) != 1:
                                if words[-len(queryPiece):] == queryPiece :
                                    words = words[:-len(queryPiece)]
                                    wordtxt = queryPhrase
                            words.append(wordtxt)

                dps.append(word['dp_label'])
                morph.append(word['morph_ev'])

            times = [] 
            for time in sentence['time'] :
                rt_date = time['rt_date']
                if len(rt_date) < 5 :
                    rt_date += '-00-00'
                elif len(rt_date) < 8 :
                    rt_date += '-00'
                times.append(rt_date)

            events = []
            for event in sentence['event'] :
                events.append(event)

            for time in times :
                for event in events :
                    if time in timelineSet :
                        if event in timelineSet[time] :
                            if 'tlink' not in sentence :
                                sentenceSets[time].append([sentence['sentence'], morph, words, [], dps])
                            else :
                                sentenceSets[time].append([sentence['sentence'], morph, words, sentence['tlink'], dps])
                            break

    ### Main Phase 1. get keyword
    totalFreqOfWords = {} # for idf
    for date, sents in sentenceSets.items():
        for sent in sents :
            words = sent[2]
            for word in words :
                word = isEnglish(word)
                if word not in totalFreqOfWords :
                    totalFreqOfWords[word] = 0

    ### compute idf
    for date, sents in sentenceSets.items():
        appearWords = []
        for sent in sents :
            words = sent[2]
            
            for word in words :
                word = isEnglish(word)
                appearWords.append(word)

        for word in totalFreqOfWords :
            word = isEnglish(word)
            if word in appearWords :
                totalFreqOfWords[word] += 1

    # compute tf
    tfScores = {}
    for date, sents in sentenceSets.items():
        tfScores[date] = []
        freqOfTotalWord = 0
        freqOfWords = {}
        curtfscores = {}
        for sent in sents :
            words = sent[2]
            for word in words :
                word = isEnglish(word)
                if word not in freqOfWords :
                    freqOfWords[word] = 0
                freqOfWords[word] += 1
                freqOfTotalWord += 1

        for word in freqOfWords :
            word = isEnglish(word)
            tf = freqOfWords[word] / freqOfTotalWord
            idf = len(totalFreqOfWords) / totalFreqOfWords[word] 

            tfidf = tf * idf

            curtfscores[word] = tfidf
            


        import operator
        curtfscores = dict(sorted(curtfscores.items(), key=operator.itemgetter(1), reverse = True))

        keywords, i = [], 0
        for keyword in curtfscores :
            keywords.append(keyword)
            i += 1
            if i == 10 :
                break

        tfScores[date].append(keywords)
        
    return tfScores, sentenceSets

