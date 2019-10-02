# -*- coding: utf-8 -*-
import os, sys, json
import re
import shutil

import collections
from collections import OrderedDict

def isEnglish(input_s):
    convert_s = ''
    for input_c in input_s:	
        if ord('a') <= ord(input_c.lower()) <= ord('z') :
            convert_s += input_c.lower()
        else :
            convert_s += input_c

    return convert_s

def saveCand(timelinePath, candidateSets) :
    candPath = timelinePath + 'CandSents.json'
    with open(candPath, 'w', encoding = 'utf-8') as f:
        json.dump(candidateSets, f)

    ### readable
    candPath = timelinePath + 'readableCandidates.txt'
    with open(candPath, 'w', encoding = 'utf-8') as f:
        for date in candidateSets :
            for sentence, info in candidateSets[date].items() :
                f.write(date+'\t'+sentence+'\n')

    ### for morph analysis
    candPath = timelinePath + 'candidate_origin/'
    candMorphPath = timelinePath + 'candidate_morph/'

    if os.path.exists(candPath) :
        shutil.rmtree(candPath)
        shutil.rmtree(candMorphPath)

    if not os.path.exists(candPath) :
        os.makedirs(candPath)
        os.makedirs(candMorphPath)

    for date in candidateSets :
        i = 0
        for sentence, info in candidateSets[date].items() :
            fname = date + '_' + str(i) + '.txt'
            info['sentence'] = sentence
            with open(candPath + fname, 'w', encoding = 'utf-8') as f:
                json.dump(info, f, ensure_ascii = False)

            with open(candMorphPath + fname, 'w', encoding = 'utf-8') as f:
                json.dump({"morph" :info['morphs'], "dependency":info['dependency']}, f)
            i += 1

def getKeyword(keywords, sentenceSets, queryWordSet, threshold, alpha) :
    candidateSets = {}
    candidateInfo = {}

    for date, sents in sentenceSets.items() :

        candidates = {}
        scoreSet = []
        tsKeywords = keywords[date][0] # get keyword in this date

        for sent in sents :
            sentence = sent[0] # sentence
            morphs = sent[1] # morph set
            words = sent[2] # keyword set
            relation = sent[3] # relation set
            dependency = sent[4] # dependency information
            #print(sent)
            sentence = re.sub(' +', ' ', sentence)
            if len(relation) == 0 :
                continue
            #else :
            #	print(sent) 
            #	sys.exit(1)

            spacing = len(sentence.split())

            if threshold != -1 and spacing > threshold :
                continue

            inforScore = 0 
            for key in tsKeywords :
                if key in words :
                    if key in queryWordSet :
                        _score = alpha + 1
                        inforScore += _score
                    else :
                        inforScore += 1
            _scoreSet = scoreSet.copy()
            _scoreSet = list(set(_scoreSet))

            totalnumOftopScore = 0
            minScore = -1
            swt = 0

            while(1):
                if len(_scoreSet) == 0 :
                    break
                maxScore = max(_scoreSet)
                totalnumOftopScore += scoreSet.count(maxScore)
                if totalnumOftopScore > 10 :
                    swt = 1
                    minScore = maxScore
                    break
                _scoreSet.remove(maxScore)

                if _scoreSet is None :
                    break

            if swt == 1 and inforScore < minScore :
                continue

            # 문장 개수 조절할 수 있는 Threshold. 늘이면 속도 증가
            if len(candidates) < 5 :
                candidates[sentence] = {'inforScore' : inforScore, 'morphs' : morphs, 'keyword' : words, 'relation' : relation, 'dependency' : dependency}
                scoreSet.append(inforScore)

            else :
                if inforScore > minScore :
                    delList = []
                    for key, value in candidates.items() :
                        if value['inforScore'] == minScore :
                            delList.append(key)

                    for key in delList :
                        del candidates[key]
                    candidates[sentence] = {'inforScore' : inforScore, 'morphs' : morphs, 'keyword' : words, 'relation' : relation, 'dependency' : dependency}
                    scoreSet.append(inforScore)
                elif inforScore == minScore :
                    candidates[sentence] = {'inforScore' : inforScore, 'morphs' : morphs, 'keyword' : words, 'relation' : relation, 'dependency' : dependency}
                    scoreSet.append(inforScore)

        candidateSets[date] = candidates

    return candidateSets

def main(tfScores, sentenceSets, timelinePath, query, threshold = 20, alpha = 0.5):
    query = isEnglish(query)
    queryWordSet = query.split()

    candidateSets = getKeyword(tfScores, sentenceSets, queryWordSet, threshold, alpha)

    saveCand(timelinePath, candidateSets)
