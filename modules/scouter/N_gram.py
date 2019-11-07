'''
/**************************************************
 * Copyright (c) 2019 KNU DKE Lab. To Present
 * All rights reserved.
 **************************************************/

 extract n-gram from newspaper documents for topic extracting

'''
from collections import Counter

def ngram(data):
    #input: analyzed_text {'sentence': {}}
    #output: uni, bi, trigram counter : e.g) ('sentence' : 5)
    text = []
    remove_word_list = ['지난','해','이날','오전','내년','달','얘기','지난해','지난달','오후','오전','량','일']

    for i in range(len(data)):
        paragraph = []
        for sen in data[i]['sentence']:
            paragraph.append(sen)
        text.append(paragraph)

    total_noun_list = []
    unigram_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()

    for p in text:
        noun_list = []
        for s in p:
            noun_sentence = [doc['lemma'] for doc in s['morp'] if doc['type'] in ['NNG', 'NNP']]
            new_noun_sentence = []
            for noun in noun_sentence:
                if noun not in remove_word_list: new_noun_sentence.append(noun)
            noun_list.append(new_noun_sentence)
            unigram_counter.update(new_noun_sentence)
        total_noun_list.append(noun_list)

    for noun_list in total_noun_list:
        for noun_sentence in noun_list:
            if len(noun_sentence) < 2: continue
            for i in range(1, len(noun_sentence)):
                # append bigram in bigram list(counter)
                bigram = noun_sentence[i-1] + ' ' + noun_sentence[i]
                bigram_counter.update({bigram: 1})
            if len(noun_sentence) < 3: continue
            for i in range(2, len(noun_sentence)):
                trigram = noun_sentence[i-2] + ' ' + noun_sentence[i-1] + ' ' +noun_sentence[i]
                trigram_counter.update({trigram: 1})

    prune(unigram_counter)
    prune(bigram_counter)
    prune(trigram_counter)

    return unigram_counter, bigram_counter, trigram_counter



def prune(ngram_counter):
    # pruning words in n-gram
    junk_word_list = []

    for ngr in ngram_counter:
        # add a word its counter is 1 or length of word is 1 to junk list
        if ngram_counter[ngr] < 2 or len(ngr) < 2: junk_word_list.append(ngr)
    for junk_word in junk_word_list:
        # delete words
        del ngram_counter[junk_word]

