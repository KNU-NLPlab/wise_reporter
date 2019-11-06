# -*- coding: utf-8 -*-
# Step A. makeTimeline
import os, sys, json

import collections
from collections import OrderedDict

import peakutils

import numpy as np


def collectTimes(doc_save_list) :
    timelineSet, publicTimelineSet = {}, {}
    collectDates = set()
    
    ### Phase 1. collect all event - temporal expression pair 
    for tmpdic in doc_save_list :

        sentences = tmpdic['extract']
        publicDate = tmpdic['writetime']
        collectDates.add(publicDate)

        # Collect Published Date(for burst point analysis)
        if publicDate not in publicTimelineSet :
            publicTimelineSet[publicDate] = 1
        else :
            publicTimelineSet[publicDate] += 1

        # Collect timestamp(for timeline analysis)
        for idx, sentence in enumerate(sentences) :
            if 'tlink' not in sentence :
                continue
            elif len(sentence['tlink']) == 0 :
                continue
            #if len(sentence['time']) != 1:
            #	continue
            tlinkSet = sentence['tlink']

            for tlinkElem in tlinkSet :
                eventWord, timeWord = tlinkElem[0]['event_lexicon'], tlinkElem[1]['rt_date']

                if timeWord not in timelineSet :
                    timelineSet[timeWord] = {}
                    timelineSet[timeWord]['timefrequency'] = 0
                    timelineSet[timeWord]['eventfrequency'] = 0
                timelineSet[timeWord]['timefrequency'] += 1

                if eventWord not in timelineSet[timeWord] :
                    timelineSet[timeWord][eventWord] = 1
                    timelineSet[timeWord]['eventfrequency'] += 1
                else :
                    timelineSet[timeWord][eventWord] += 1

            #with open(inPath + file, 'w', encoding = 'utf-8') as f:
            #	json.dump(tmpdic, f)

    collectDates = list(collectDates)
    publicTimelineSet = dict(collections.OrderedDict(sorted(publicTimelineSet.items())))

    return collectDates, publicTimelineSet, timelineSet

def getBurst(internalTimeline, publicTimelineSet, threshold):
    freqOfTimes, freqOfEvents = [0], [0]
    dateOfTimes = [0]

    temp = []
    for key, value in internalTimeline.items():
        totalEvents = 0 

        for eventName, eventFreq in value.items() :
            if 'frequency' in eventName :
                continue
            else :
                totalEvents += eventFreq
        freqOfTimes.append(totalEvents)
        freqOfEvents.append(value['eventfrequency'])
        dateOfTimes.append(key)
        temp.append(key)

    freqOfTimes = np.array(freqOfTimes, dtype = int)
    freqOfEvents = np.array(freqOfEvents, dtype = int)

    numOfTime = len(freqOfTimes)

    freqOfPublic = [0]
    dateOfPublic = [0]

    for key, value in publicTimelineSet.items():
        freqOfPublic.append(value)
        dateOfPublic.append(key)

    freqOfPublic = np.array(freqOfPublic, dtype = int)
    burst = peakutils.indexes(freqOfPublic, thres = threshold, min_dist = 1)

    numOfburst = len(burst)
    burstTimeSet = [] 
    for index in burst :
        burstTimeSet.append(dateOfPublic[index])

    return burstTimeSet

def seperateTimeline(timelineSet, collectDates) :
    internalTimeline = {} # burst + k
    externalTimeline = {}

    ### Seperate internal/external timestamp
    for key, value in timelineSet.items() :
        if len(key) < 5 :
            key += '-00-00'
        elif len(key) < 8 :
            key += '-00'

        if key not in collectDates : # if timestamp is not in collected days
            externalTimeline[key] = value
        else :
            internalTimeline[key] = value

    externalTimeline = dict(collections.OrderedDict(sorted(externalTimeline.items())))
    internalTimeline = dict(collections.OrderedDict(sorted(internalTimeline.items())))

    return externalTimeline, internalTimeline

def makeTimeline(externalTimeline, internalTimeline, burstTimeSet) : ### Phase 2. Select Main TimeStamp ###
    alpha = 2
    beta = 0

    numOfburst = len(burstTimeSet)
    numOfinternalTS = numOfburst * alpha
    numOfexternalTS = numOfburst + beta

    TimelineSets = [internalTimeline, externalTimeline]
    numOfTSSets = [numOfinternalTS, numOfexternalTS]
    timelineSet = {}

    mode = 1
    for timeline, numOfTS in zip(TimelineSets, numOfTSSets) :
        freqOfTimes, freqOfEvents = [], []
        for key, value in timeline.items():
            freqOfTimes.append(value['timefrequency'])
            freqOfEvents.append(value['eventfrequency'])

        maxTimeFreq = max(freqOfTimes)
        maxEventFreq = max(freqOfEvents)

        # Normalize with time and event
        for timestamp, events in timeline.items() :
            timeline[timestamp]['timeScore'] = events['timefrequency'] / maxTimeFreq
            timeline[timestamp]['eventScore'] = events['eventfrequency'] / maxEventFreq
            timeline[timestamp]['TSI'] = timeline[timestamp]['timeScore'] * timeline[timestamp]['eventScore']

        # Sort by TSI
        timeline = OrderedDict(sorted(timeline.items(), key=lambda kv: kv[1]['TSI'], reverse=True))

        # hightest top k 
        idx = 0
        burst_idx = 0
        if mode == 1 : ## internal time
            numOfETC = numOfTS - numOfburst
            for timestamp, event in timeline.items() :
                if idx == numOfETC and burst_idx == numOfburst :
                    break
                if timestamp in burstTimeSet : # add burst timestamp timelineSet
                    timelineSet[timestamp] = event
                    timelineSet[timestamp]['burst'] = True
                    burst_idx += 1
                elif idx < numOfETC : # add m timestamp
                    timelineSet[timestamp] = event
                    timelineSet[timestamp]['burst'] = False
                    idx += 1
            mode = 2

        else : ## external time
            for timestamp, event in timeline.items() :
                if idx == numOfTS :
                    break
                timelineSet[timestamp] = event
                timelineSet[timestamp]['burst'] = False
                idx += 1

    # sort by time
    timelineSet = dict(collections.OrderedDict(sorted(timelineSet.items())))

    return timelineSet

def main(doc_save_list, threshold=0.3, alpha=2, beta=0) :

    # Step 1. collectTimes
    collectDates, publicTimelineSet, timelineSet = collectTimes(doc_save_list)

    # Step 2. Seperate external/internal
    externalTimeline, internalTimeline = seperateTimeline(timelineSet, collectDates)

    # Step 3. Get burstpoint
    burstTimeSet = getBurst(internalTimeline, publicTimelineSet, threshold)
    numOfburst = len(burstTimeSet)
    
    # Step 4. Make Timeline
    timelineSet = makeTimeline(externalTimeline, internalTimeline, burstTimeSet)

    return publicTimelineSet, burstTimeSet, timelineSet
    # save timeline
    #with open(outPath + 'timeline.json', 'w', encoding = 'utf-8') as f:
    #    json.dump(timelineSet, f, ensure_ascii = False)

    """
    with open(outPath + 'readableTimeline.txt', 'w', encoding = 'utf-8') as f:
        for timeStamp in timelineSet :
            #print(timeStamp)
            events = timelineSet[timeStamp]
            eventFreqs = len(events) - 1

            f.write(timeStamp + '\t' + str(events['timefrequency']) + '\t' + str(events['eventfrequency']) + '\n')
    """

