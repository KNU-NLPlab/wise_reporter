# -*- coding: utf-8 -*-

import json

import onmt
import onmt.io
import torch
import torch.cuda
from torch.autograd import Variable

from rouge import Rouge 

class Reward():
    def __init__(self, reward):
        self.reward = reward
        print("Reward : {}".format(reward))
        if reward == "rouge":
            self.rouge = Rouge(metrics=["rouge-l"], stats=["f"])
        elif "entailment" in reward:
#            entail_model_path = "0617_entail_400k_model.tar.gz" # kor entail index:1
            entail_model_path = "allennlp_eng_model.tar.gz" # eng entail index:0, cotra, neu:1,2
            self.entailment = onmt.modules.Entailment_scorer(entail_model_path, 0) # model  path and index of entailment
        self.eps = -1e-5
        self.error = 0
        
    def get_rouge_reward(self, batch, sample_indices, max_indices, copy):
#         print("Reward line:5 sample indices", sample_indices) # tgt_len * batch
#         print("Reward line:5 max indices", max_indices) # tgt_len * batch
#         print("rweard line:7 sample_indeices[:,0]", sample_indices[:,0])
#         print("rweard line:20 batch", batch)
#         print("rweard line:21 batch", batch.tgt)
        
        tgt_vocab = batch.dataset.fields['tgt'].vocab
        global_vocab = batch.dataset.fields['src'].vocab
#         print("batch src")
#         print(batch.src)
#         input()
             
        
        sample_scores = []
        max_scores = []
        alignments = [] # for efficiency, calculate alignments here
        
        for i in range(len(batch)):
            in_batch_index = batch.indices.data[i]
#             print("Reward line:11 in batch index",in_batch_index)
#             print("Reward line:29 in raw example", len(batch.dataset.examples))
#             print("Reward line:30 batch dataset fileds tgt", batch.dataset.fields['tgt'])
            src_vocab = batch.dataset.src_vocabs[in_batch_index] if copy else global_vocab
            
#             raw_tgt = batch.dataset.examples[in_batch_index].tgt
            raw_tokens = self.build_target_tokens(src_vocab, tgt_vocab, batch.tgt.data[1:,i])
#             print("reward line:36 raw_tgt", raw_tgt)
#             print("reward line:37 raw_tokens", raw_tokens)
            sample_tokens = self.build_target_tokens(src_vocab, tgt_vocab, sample_indices[:,i])
            max_tokens = self.build_target_tokens(src_vocab, tgt_vocab, max_indices[:,i])
#             print("reward line:16 sample tokens",sample_tokens)
            sample_rouge_f1_s = self.calculate_rouge(sample_tokens, raw_tokens)
            max_rouge_f1_s = self.calculate_rouge(max_tokens, raw_tokens)
            
            # calculate alginemts
            mask = [0] + [src_vocab.stoi[w] for w in sample_tokens]
            alignments.append(mask)
            
#             print("reward line:37 sample_tokens", sample_tokens)
#             print("reward line:37 max_tokens", max_tokens)
        
            sample_scores.append(sample_rouge_f1_s['rouge-l']['f'])
            max_scores.append(max_rouge_f1_s['rouge-l']['f'])
            
            if torch.rand(1)[0] <= 0.005:
                src_tokens = self.build_target_tokens(src_vocab, batch.dataset.fields['src'].vocab, batch.src[0].data[:,i])
                print("in batch index = {}".format(in_batch_index))
                print("\t src tokes")
                print("\t\t", src_tokens)
                print("\t target tokens")
                print("\t\t", raw_tokens)
                print("\tsampled tokens")
                print("\t\t", sample_scores[-1], sample_tokens)
                print("\t max tokens")
                print("\t\t", max_scores[-1], max_tokens)            
#         print("Rewards line:72 alignments", alignments )
        
        max_sample_len = max(len(x) for x in alignments)
        max_sample_len = max(sample_indices.size(0)+1, max_sample_len)
#         print("Reward line:75 sample_indices", sample_indices.size())        
#         print("Reward line:76 max", max(len(x) for x in alignments))
        for i in range(len(alignments)):
            alignments[i] += [0] * max(0, max_sample_len - len(alignments[i]))
            alignments[i] = torch.LongTensor(alignments[i]).cuda()
#         print("Rewards line:77 alignments", alignments )            
        sample_alignments = torch.stack(alignments).transpose(0,1)
#             print("reward line:29 rouge", sample_rouge_f1_s, max_rouge_f1_s)
        sample_scores = torch.Tensor(sample_scores).cuda()
        max_scores = torch.Tensor(max_scores).cuda()
        batch_scores = max_scores - sample_scores
        return batch_scores, sample_scores, max_scores, sample_alignments
    
    def get_entailment_reward(self, batch, sample_indices, max_indices, entail_type):
#         print("Reward line:5 sample indices", sample_indices) # tgt_len * batch
#         print("Reward line:5 max indices", max_indices) # tgt_len * batch
#         print("rweard line:7 sample_indeices[:,0]", sample_indices[:,0])
#         print("rweard line:20 batch", batch)
#         print("rweard line:21 batch", batch.tgt)
        
        tgt_vocab = batch.dataset.fields['tgt'].vocab
        src_vocab = batch.dataset.fields['src'].vocab
#         print("batch src")
#         print(batch.src)
#         input()
             
        
        sample_scores = []
        max_scores = []
        alignments = [] # for efficiency, calculate alignments here
        
        for i in range(len(batch)):
            in_batch_index = batch.indices.data[i]
#             print("Reward line:11 in batch index",in_batch_index)
#             print("Reward line:29 in raw example", len(batch.dataset.examples))
#             print("Reward line:30 batch dataset fileds tgt", batch.dataset.fields['tgt'])
#            src_vocab = batch.dataset.src_vocabs[in_batch_index]
        
#             raw_tgt = batch.dataset.examples[in_batch_index].tgt
            raw_src_tokens = self.build_src_tokens(src_vocab, batch.src[0].data[:,i])
#             print("reward line:36 raw_tgt", raw_tgt)
#             print("reward line:37 raw_tokens", raw_tokens)
            sample_tokens = self.build_target_tokens(src_vocab, tgt_vocab, sample_indices[:,i])
            max_tokens = self.build_target_tokens(src_vocab, tgt_vocab, max_indices[:,i])
#             print("reward line:16 sample tokens",sample_tokens)
#            sample_entail_s = self.get_entailment_score(raw_src_tokens, sample_tokens)
#            max_entail_s = self.get_entailment_score(raw_src_tokens, max_tokens)
            
            # calculate alginemt
            instance_src_vocab = batch.dataset.src_vocabs[in_batch_index]
            raw_tokens = self.build_target_tokens(instance_src_vocab, tgt_vocab, batch.tgt.data[1:,i])

            if entail_type == "entailment_src_hyp_sample":
              hyp_tokens = raw_src_tokens
            elif entail_type == "entailment_src_hyp_gold":
              hyp_tokens = raw_src_tokens
              max_tokens = raw_tokens
            elif entail_type == "entailment_tgt_hyp":
              hyp_tokens = raw_tokens
            else:
              input("Parameter Error!")             

            sample_entail_s = self.get_entailment_score(hyp_tokens, sample_tokens, True)
            max_entail_s = self.get_entailment_score(hyp_tokens, max_tokens, True) # use gold target to baseline


            mask = [0] + [instance_src_vocab.stoi[w] for w in sample_tokens]
            alignments.append(mask)
            
#            print("reward line:37 sample_tokens", sample_entail_s, sample_tokens)
#            print("reward line:37 max_tokens", max_entail_s, max_tokens)
        
            sample_scores.append(sample_entail_s)
            max_scores.append(max_entail_s)
            
            if torch.rand(1)[0] <= 0.005:
#                 src_tokens = self.build_target_tokens(src_vocab, batch.dataset.fields['src'].vocab, batch.src[0].data[:,i])
                src_tokens = raw_src_tokens
                print("in batch index = {}".format(in_batch_index))
                print("\t src tokes")
                print("\t\t", src_tokens)
                print("\t target tokens")
                print("\t\t", raw_tokens)
                print("\tsampled tokens")
                print("\t\t", sample_scores[-1], sample_tokens)
                print("\t max tokens")
                print("\t\t", max_scores[-1], max_tokens)            
#         print("Rewards line:72 alignments", alignments )
        
        max_sample_len = max(len(x) for x in alignments)
        max_sample_len = max(sample_indices.size(0)+1, max_sample_len)
#         print("Reward line:75 sample_indices", sample_indices.size())        
#         print("Reward line:76 max", max(len(x) for x in alignments))
        for i in range(len(alignments)):
            alignments[i] += [0] * max(0, max_sample_len - len(alignments[i]))
            alignments[i] = torch.LongTensor(alignments[i]).cuda()
#         print("Rewards line:77 alignments", alignments )            
        sample_alignments = torch.stack(alignments).transpose(0,1)
#             print("reward line:29 rouge", sample_rouge_f1_s, max_rouge_f1_s)
        sample_scores = torch.Tensor(sample_scores).cuda()
        max_scores = torch.Tensor(max_scores).cuda()
        batch_scores = max_scores - sample_scores
        return batch_scores, sample_scores, max_scores, sample_alignments        
                       
    
    def get_batch_reward(self, batch, sample_indices, max_indices, copy=None):
        if self.reward == "rouge":
            assert copy is not None
            return self.get_rouge_reward(batch, sample_indices, max_indices, copy)
        elif "entailment" in self.reward:
            return self.get_entailment_reward(batch, sample_indices, max_indices, self.reward)
        
        
    def get_entailment_score(self, src_tokens, sample_tokens, length_penalty=False):
        premise = " ".join(src_tokens)
        hypothesis = " ".join(sample_tokens)
        
        json_data = {"premise":premise, "hypothesis":hypothesis}
#        json_data = json.dumps(json_data, ensure_ascii=False)
#        print(json_data)
        
        score = self.entailment.predict_entailment(json_data)
        if length_penalty:
          penalty = len(sample_tokens) / len(src_tokens) 
          score = penalty * score
        
        return score
        
               

            
    def calculate_rouge(self, hyp, ref):
        hyp = " ".join(hyp)
        ref = " ".join(ref)
        
        score = self.rouge.get_scores(hyp, ref)
        return score[0]
        
        
    def build_src_tokens(self, src_vocab, indices):
        tokens = []
#         print("reward line:18 onmt.io.EOS_WORD", onmt.io.EOS_WORD)
        for tok in indices:
            try:
                tokens.append(src_vocab.itos[tok])                
            except IndexError:
                self.error += 1
                print("Reward line 82: Error index occured {}".format(self.error))
                tokens.append('<unk>')
        return tokens     
    

    def build_target_tokens(self, src_vocab, tgt_vocab, pred):
        tokens = []
#         print("reward line:18 onmt.io.EOS_WORD", onmt.io.EOS_WORD)
        for tok in pred:
            try:
                if tok < len(tgt_vocab):
                    tokens.append(tgt_vocab.itos[tok])
                else:
                    tokens.append(src_vocab.itos[tok - len(tgt_vocab)])
                if tokens[-1] == onmt.io.EOS_WORD:
                    tokens = tokens[:-1]
                    break
            except IndexError:
                self.error += 1
                print("Reward line 82: Error index occured {}".format(self.error))
                tokens.append('<unk>')
        return tokens
    
    def criterion(self, input, seq, reward):
#         print("reward line 69 input", input)
#         print("reward line 69 seq", seq)
#         print("reward line 69 reward", reward)
#         print("reward line 69 reward", reward.expand_as(input))
        reward = reward.expand_as(input) + self.eps
        print("reward line 76 reward", reward)    
        def to_contiguous(tensor):
            if tensor.is_contiguous():
                return tensor
            else:
                return tensor.contiguous()    
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        print("reward line 89 input req", input.requires_grad)
#         output = - input * reward * Variable(mask)
        output = - input
        output = torch.sum(output) / torch.sum(mask) / 7
#         output = torch.sum(output)
        
        return output
