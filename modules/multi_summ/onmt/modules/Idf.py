# -*- coding : utff-8 -*-
# Idf.py
# 18.07.03
# code for idf (inverse document freq)
# src file path is hard coded

from math import log

class Idf():
    def __init__(self, src_file_path=None, revision_num=1):
        if src_file_path is None:
            self.src_file_path = "article_data/article_src_train_src_500_tar_50.txt"
            self.src_file_path = "article_data/article_src_train_src_500_tar_50_2.txt"
            
        else:
            self.src_file_path = src_file_path
        self.revision_num = revision_num # revision num when revision true, it reviise idf values that below revision_num to reivison_num
    
        # get idf value
#     words = [ fields["tgt"].vocab.itos[i] for i in range(len(fields["tgt"].vocab)) ]
    def get_df(self, src_file_path, words):
        words_df = [0] * len(words)
        
        if src_file_path is None:
            src_file_path = self.src_file_path
        print("Idf line:26 src file path : {}".format(src_file_path))
        with open(src_file_path, 'r', encoding="utf-8") as src_file:
            import collections
            
            cnt = 0
            for line in src_file:
                cnt += 1
                src_words = line.strip().split()
                src_words_dict = { word:1 for word in src_words }
                for i in range(len(words)):
                    if words[i] in src_words_dict:
                        words_df[i] += 1
                if cnt % 10000 == 0:
                    print("Idf line:36 current cnt: {}".format(cnt))
        return words_df, cnt
    
    
    def get_idf_weights(self, src_file_path, words, special_symbol_index=[], revision=False):
        words_df, cnt = self.get_df(src_file_path, words)
        print("Idf line:37 complete get df information")
        
        words_df_weights = [0] * len(words_df)
   
        for i in range(len(words)):
            word = words[i]
            if words_df[i] == 0:
#                 words_df[i] = 1
                if i not in special_symbol_index:
                    special_symbol_index.append(i) # give unk word to default weight
                words_df_weights[i] = round(log( cnt / 1 ), 2)
            else:
                words_df_weights[i] = round(log( cnt / words_df[i] ), 2)
            
        print("Idf line:52 write idf info to file")
        with open("idf_info.txt", "w", encoding="utf-8") as out_file:
            for i in range(len(words)):
                print("{}\t{}\t{}".format(words[i], words_df[i], words_df_weights[i]), file=out_file)
                
        if special_symbol_index != None:
            for idx in special_symbol_index:                
                words_df_weights[idx] = 1
                
        if revision:
            revised_cnt = 0
            for i in range(len(words_df_weights)):
                weight = words_df_weights[i]
                if weight < self.revision_num:
                    words_df_weights[i] = self.revision_num
                    revised_cnt += 1
                
#             words_df_weights = [ weight if weight > self.revision_num else revision_num; revised_cnt += 1 for weight in words_df_weights]
            print("Revise weight that below {} to {}".format(self.revision_num, self.revision_num))
            print("{} revised".format(revised_cnt))
            
            
                
        return words_df_weights

