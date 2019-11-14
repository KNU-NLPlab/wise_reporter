# -*- coding: utf-8 -*-

import os, sys, json
import collections
from collections import OrderedDict
import codecs
import math


def openCandidates(timelinePath, keywordSet):
    # 1. open candidate sentence
    candPath = timelinePath + 'candidate_redundancy_origin/'

    candidateSentSets, candidateScoreSets, candidateKeywordSets, candidateRelationSets, candidateMorphTagSets = {}, {}, {}, {}, {}
    timeline = []

    for path, dir, files in os.walk(candPath):
        for file in files:
            # Load sentence
            f = open(candPath + file, 'r', encoding='utf-8')
            _sentence = json.loads(f.read())
            f.close()

            sentence, score, keywords, relation, morph_tag = _sentence['sentence'], float(_sentence['inforScore']), _sentence['keyword'], _sentence['relation'], _sentence['morph_tag']

            # Save sentence each date
            date = file.split('_')[0]
            if date not in candidateSentSets:
                candidateSentSets[date] = []
            candidateSentSets[date].append(sentence)

            if date not in candidateScoreSets:
                candidateScoreSets[date] = []
            candidateScoreSets[date].append(score / 10)

            keyword = ''

            for key in keywords:
                if keyword != '':
                    keyword += '/'
                keyword += key

            if date not in candidateKeywordSets:
                candidateKeywordSets[date] = []
            candidateKeywordSets[date].append(keyword)

            if date not in candidateRelationSets:
                candidateRelationSets[date] = []
            candidateRelationSets[date].append(relation)

            if date not in candidateMorphTagSets:
                candidateMorphTagSets[date] = []
            candidateMorphTagSets[date].append(morph_tag)

            if date not in timeline:
                timeline.append(date)
    timeline.sort()
    return candidateSentSets, candidateScoreSets, candidateKeywordSets, candidateRelationSets, candidateMorphTagSets, timeline


def main(keywordSet, inPath, outPath, timelineSet, burstTimeSet, query, threshold=3):
    timelinePath = inPath + '/'

    querySet = query.split()

    # A. Open Candidate Sentences
    candidateSentSets, candidateScoreSets, candidateKeywordSets, candidateRelationSets, candidateMorphTagSets, timeline = openCandidates(
        timelinePath, keywordSet)

    # B. grammar role
    # SBJ : subject, OBJ = object

    candPath = inPath + 'candidate_redundancy_morph/'
    candidateMorphSets = {}
    candidateInfoSets = {}

    for path, dir, files in os.walk(candPath):
        for file in files:
            # load sentence
            morphAnalysis = {}

            with open(candPath + file, 'r', encoding='utf-8') as f:
                morphAnalysis = json.loads(f.read())

            morphState = [0, 0, 0, 0]  # sub, obj, pre, absent
            morph_result = morphAnalysis['morph']
            dp_result = morphAnalysis['dependency']

            for idx, morph in enumerate(morph_result):
                morphs = morph.split('+')
                for mor in morphs:
                    wordtxt = mor.split('/')[0]
                    mor = mor

                    if wordtxt not in querySet:
                        continue

                    dp = dp_result[idx]
                    if 'SBJ' in dp:
                        morphState[0] = 1
                    elif 'OBJ' in dp:
                        morphState[1] = 1
                    else:
                        morphState[2] = 1

            if morphState == [0, 0, 0, 0]:
                morphState = [0, 0, 0, 1]

            # save grammar result each date
            date = file.split('_')[0]
            if date not in candidateMorphSets:
                candidateMorphSets[date] = []
                candidateInfoSets[date] = []
            candidateMorphSets[date].append(morphState)
            candidateInfoSets[date].append(morph_result)

    numSentSet = []
    for key in candidateSentSets:
        numSentSet.append(len(candidateSentSets[key]))

    # 3. select sentence
    SelectedSent = []

    numSentSet = []
    for key in candidateSentSets:
        numSentSet.append(len(candidateSentSets[key]))

    totalNumOfPair = 0
    for i in range(len(numSentSet) - 1):
        numofpair = numSentSet[i] * numSentSet[i + 1]
        totalNumOfPair += numofpair

    from operator import add
    transition = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    totalPair = 0
    for idx in range(len(timeline) - 1):
        date_i, date_j = timeline[idx], timeline[idx + 1]
        sentences_i, sentences_j = candidateSentSets[date_i], candidateSentSets[date_j]
        morphStates_i, morphStates_j = candidateMorphSets[date_i], candidateMorphSets[date_j]

        for idx_i, sent_i in enumerate(morphStates_i):
            morph_i = morphStates_i[idx_i].index(1)
            _transition = [0, 0, 0, 0]
            for idx_j, sent_j in enumerate(morphStates_j):
                morph_j = morphStates_j[idx_j]
                _transition = list(map(add, _transition, morph_j))
                totalPair += 1

            transition[morph_i] = list(map(add, transition[morph_i], _transition))

    epsilon = sys.float_info.epsilon
    log_transition = []

    totalLogPair = 0
    maxLog = 0
    for a in transition:
        # _transitionProb = [x / totalNumOfPair for x in a]
        _log_transition = []
        for idx, x in enumerate(a):
            x += 1
            if idx == 3:
                x = 1 / x

            logTrans = math.log(x + epsilon)
            if logTrans > maxLog:
                maxLog = logTrans
            totalLogPair += logTrans
            _log_transition.append(logTrans)
        log_transition.append(_log_transition)

    logTransProb = []
    for a in log_transition:
        _log_transition = []
        for idx, x in enumerate(a):
            _log_transition.append(x / maxLog)
        # print(_log_transition)
        logTransProb.append(_log_transition)

    Score = logTransProb

    ###### Start Summarization : Used BeamSearch ######
    # Start to first node

    startDate = timeline[0]
    startScore = candidateScoreSets[startDate].copy()
    topScore = startScore.copy()
    topScore.sort(reverse=True)

    if len(startScore) < threshold:
        topScore = topScore[-1]
    else:
        topScore = topScore[threshold - 1]

    pathList, scoreList, lastList = [], [], []

    for st_idx, st_score in enumerate(startScore):
        if st_score < topScore:
            continue
        pathList.append([st_idx])
        scoreList.append(st_score)
        lastList.append(st_idx)

    # calculate next nodes
    print(" >> phase 4. calculate next nodes")
    print("timeline")
    print(timeline)
    date_idx = 0
    while (1):
        if len(timeline) == 1:
            break
        date_i, date_j = timeline[date_idx], timeline[date_idx + 1]
        sentence_i, sentence_j = candidateSentSets[date_i], candidateSentSets[date_j]
        infoscore_i, infoscore_j = candidateScoreSets[date_i], candidateScoreSets[date_j]  # j is only used
        morphStates_i, morphStates_j = candidateMorphSets[date_i], candidateMorphSets[date_j]

        ### 현재의 점수 구하기
        # 점수 coherence + info
        curScoreList = []
        for idx_i in lastList:
            morph_i = morphStates_i[idx_i].index(1)
            temp = []
            for idx_j, sent_j in enumerate(morphStates_j):
                morph_j = morphStates_j[idx_j].index(1)
                _cohscore = Score[morph_i][morph_j]
                _infoScore = infoscore_j[idx_j]
                _score = _cohscore + _infoScore
                # print('morphstate j')
                temp.append(_score)
            curScoreList.append(temp)

        #### 이전까지의 top-n best path와 현재 문장들간의 점수 측정
        candScore = []
        lastElem = 0
        for pathElem, scoreElem in zip(pathList, scoreList):
            # pathList : 현재까지의 best path 경로 저장, scoreList : pathList의 각 path의 점수 저장
            curScoreSet = curScoreList[lastElem]
            for curScore in curScoreSet:
                # print(scoreElem, curScore)
                candScore.append(scoreElem + curScore)
            lastElem += 1

        #### 새로운 top-n best path 구하기
        topScore = candScore.copy()
        topScore.sort(reverse=True)
        if len(topScore) < threshold:
            topScore = topScore[-1]
        else:
            topScore = topScore[threshold - 1]

        numOfPrevPath = len(morphStates_j)

        # 새 best path & score 저장
        newPathList, newScoreList, newLastList = [], [], []
        for candIdx, candscore in enumerate(candScore):
            if candscore < topScore:
                continue

            prevIdx = int(candIdx / numOfPrevPath)
            curIdx = int(candIdx % numOfPrevPath)
            newPathElem = pathList[prevIdx].copy()
            newPathElem.append(curIdx)

            newPathList.append(newPathElem)
            newScoreList.append(candscore)
            newLastList.append(curIdx)

        pathList, scoreList = newPathList.copy(), newScoreList.copy()
        lastList = newLastList.copy()

        date_idx += 1
        print("date_idx: {}/{}".format(date_idx, len(timeline)))

        if date_idx == len(timeline) - 1:
            break

    # Get highest
    maxScoreIdx = scoreList.index(max(scoreList))
    pathList = pathList[maxScoreIdx]
    # print(pathList)

    #### Make beamsearch file ####
    print(" >> phase 4. Make beamsearch file")
    date_idx = 0
    demoResult = []

    saveScoreList = []
    startDate = timeline[date_idx]
    startSet = {}
    startSet['date'] = startDate.replace('-00', '')
    startSet['candidates'] = []

    for curIdx in range(len(candidateScoreSets[startDate])):
        temp = {}
        temp['idx'] = curIdx
        temp['sent'] = candidateSentSets[startDate][curIdx]
        temp['score'] = candidateScoreSets[startDate][curIdx]
        if curIdx == pathList[0]:
            temp['edge'] = True
        else:
            temp['edge'] = False

        startSet['candidates'].append(temp)

    demoResult.append(startSet)
    saveScoreList.append(candidateScoreSets[startDate])

    while (1):
        if len(timeline) == 1:
            break
        date_i, date_j = timeline[date_idx], timeline[date_idx + 1]
        sentence_i, sentence_j = candidateSentSets[date_i], candidateSentSets[date_j]
        infoscore_i, infoscore_j = candidateScoreSets[date_i], candidateScoreSets[date_j]  # j is only used
        morphStates_i, morphStates_j = candidateMorphSets[date_i], candidateMorphSets[date_j]

        ansIdx = pathList[date_idx]
        lastScore = saveScoreList[-1][ansIdx]

        curScoreList = []
        morph_i = morphStates_i[ansIdx].index(1)
        for idx_j, sent_j in enumerate(morphStates_j):
            morph_j = morphStates_j[idx_j].index(1)
            _cohscore = Score[morph_i][morph_j]
            _infoScore = infoscore_j[idx_j]
            _score = _cohscore + _infoScore
            _score = lastScore + _score
            curScoreList.append(_score)
        saveScoreList.append(curScoreList)

        ### Get Score
        curSet = {}
        curSet['date'] = date_j.replace('-00', '')
        curSet['candidates'] = []

        for curIdx in range(len(candidateScoreSets[date_j])):
            temp = {}
            temp['idx'] = curIdx
            temp['sent'] = sentence_j[curIdx]
            temp['score'] = float('{0:.4f}'.format(curScoreList[curIdx]))
            if curIdx == pathList[date_idx + 1]:
                temp['edge'] = True
            else:
                temp['edge'] = False

            curSet['candidates'].append(temp)

        demoResult.append(curSet)

        date_idx += 1

        if date_idx == len(timeline) - 1:
            break

    savedResult = []

    for idx, value in enumerate(pathList):
        date = timeline[idx]
        keywords = candidateKeywordSets[date][value]

        temp = {}
        keyword = ''

        for key in keywordSet[date][0]:
            if keyword != '':
                keyword += '/'
            keyword += key
        temp['keyword'] = keyword
        temp['article'] = [{'sentence': candidateSentSets[date][value]}, {'morph_tag': candidateMorphTagSets[date][value]}, {'score': candidateScoreSets[date][value]}]

        if timelineSet[date]['burst'] == True:
            temp['burst'] = True
        else:
            temp['burst'] = False

        temp['morph'] = candidateInfoSets[date][value]
        temp['tlink'] = candidateRelationSets[date][value]
        if '-00' in date:
            date = date.replace('-00', '')
        temp['date'] = date
        # temp['morph'] = candidateMorphSets[date][value]
        # temp['tlink'] = candidateRelationSets[date][value]

        savedResult.append(temp)

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(savedResult, f, ensure_ascii=False)
    #    with open(os.path.join(outPath, 'detail.json'), 'w', encoding = 'utf8') as f:
    #    with open(outPath + 'detail.json', 'w', encoding = 'utf8') as f :
    #        json.dump(demoResult, f)

    # print(savedResult)

    return savedResult, demoResult