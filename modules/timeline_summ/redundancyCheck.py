# -*- coding: utf-8 -*-

import os, sys, json
import collections
from collections import OrderedDict

import shutil
from shutil import copyfile

import re
import math
from collections import Counter

from .cleanDirectory import cleanDirectory


# 전체 문장으로 morph vector
def morph_to_index(sets):
    morph2idx = {}
    count = 0
    for date, sents in sets.items():
        for sent in sents:
            morphs = sent.split()
            for morph in morphs:
                if morph not in morph2idx:
                    morph2idx[morph] = count
                    count += 1

    return morph2idx


def cosinesimilarity(x, y):
    numerator = 0
    denominator = 0
    denominator_x = 0
    denominator_y = 0
    for xv in x:
        denominator_x = denominator_x + pow(xv, 2)
    for yv in y:
        denominator_y = denominator_y + pow(yv, 2)
    denominator = math.sqrt(denominator_x) * math.sqrt(denominator_y)

    for i in range(len(x)):
        numerator = numerator + (x[i] * y[i])
    return numerator / denominator


def main(dirPath):
    # 1. open candidate sentence
    candPath = dirPath + 'candidate_morph/'
    candidateMorphSets = {}

    for path, dir, files in os.walk(candPath):
        for file in files:
            # load sentence
            morphAnalysis = {}

            with open(candPath + file, 'r', encoding='utf-8') as f:
                morphAnalysis = json.loads(f.read())

            morphState = [0, 0, 0, 0]  # sub, obj, pre, absent
            morph_result = morphAnalysis['morph']
            # morph_result = morphAnalysis['morph']['lemma']
            # dp_result = morphAnalysis['sentence'][0]['dependency']

            mors = ''
            for morph in morph_result:
                morphset = morph.split('+')
                for mor in morphset:
                    mor = mor.split('/')[0]
                    if mors != '':
                        mors += ' '
                    mors += mor
            # save grammar result each date
            date = file.split('_')[0]
            if date not in candidateMorphSets:
                candidateMorphSets[date] = []
            candidateMorphSets[date].append(mors)

    morph2idx = morph_to_index(candidateMorphSets)
    # get_result

    refineCandidateMorphSets = {}
    for date, sents in candidateMorphSets.items():
        testSents = sents.copy()
        remainSents = []
        removeId = []

        while (1):
            if len(testSents) == 0:
                break
            saveSimIdx = []
            # make criterion vector to compare others
            criterion = testSents[-1].split()
            criterionVector = [0] * len(morph2idx)
            for crit in criterion:
                criterionVector[morph2idx[crit]] = 1

            for i in range(len(testSents) - 1):
                contrast = testSents[i].split()
                contrastVector = [0] * len(morph2idx)
                for cont in contrast:
                    contrastVector[morph2idx[cont]] = 1
                similarity = cosinesimilarity(criterionVector, contrastVector)
                if similarity < 0.6:  # need to tuning
                    saveSimIdx.append(i)
            # print(date + '\t' + str(similarity) + '\t' + testSents[-1] + '\t' + testSents[i])

            remainSents.append([testSents[-1], sents.index(testSents[-1])])
            _testSents = []
            for i in saveSimIdx:
                _testSents.append(testSents[i])
            testSents = _testSents.copy()

        refineCandidateMorphSets[date] = remainSents

    ## copy

    originPath = dirPath + 'candidate_origin/'
    candPath = dirPath + 'candidate_morph/'

    copyoriginPath = dirPath + 'candidate_redundancy_origin/'
    copycandPath = dirPath + 'candidate_redundancy_morph/'

    cleanDirectory(copyoriginPath)
    cleanDirectory(copycandPath)

    for date, sents in refineCandidateMorphSets.items():
        for sent in sents:
            fname = date + '_' + str(sent[1]) + '.txt'
            copyfile(originPath + fname, copyoriginPath + fname)
            copyfile(candPath + fname, copycandPath + fname)
