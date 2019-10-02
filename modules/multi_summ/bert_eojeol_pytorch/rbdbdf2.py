# import pickle
#
# with open('vocab.korean.rawtext2.list', encoding='utf-8') as f:
#     a = f.readlines()
#
# indices = [b.split()[0][-1] == '_' for i, b in enumerate(a)]
#
#
# asd = {}
#
# for i, tf in enumerate(indices):
#     asd[i] = tf
# pickle.dump(asd, open('segment.pkl', 'wb'))
# print()

import torch
from bert_eojeol_pytorch.src_tokenizer import tokenization
from pytorch_pretrained_bert import BertTokenizer

tokenizer2 = BertTokenizer.from_pretrained('bert-large-uncased')

tokenizer = tokenization.BertTokenizer('./vocab.korean.rawtext.list')

# itos = open('./vocab.korean.rawtext2.list', encoding='utf-8').readlines()
# bas = open('../data/korean/new_news_src_train.txt').readlines()
# sldkvn = open('../data/korean/new_news_tar_train.txt').readlines()
# bas = open('../data/korean_bert/raw_train_src_1m.txt').readlines()
# sldkvn = open('../data/korean_bert/raw_train_tgt_1m.txt').readlines()
sidn = open('../data/cnndm/train.txt.src').readlines()
wroibn = open('../data/cnndm/train.txt.tgt.tagged').readlines()

# a = torch.load('../data/korean_bert/korean_bert_subword_512220.train.0.pt')
# a = torch.load('../data/korean/new_news_trunc450.train.0.pt')
# a = torch.load('../data/cnndm/BERT_CNNDM450120.train.0.pt')
# b = torch.load('../data/korean/new_news_trunc450.vocab.pt')

c = []
# for asdv in a[:100000]:
#     src = ' '.join(asdv.src[0])
    # tgt = ' '.join(asdv.tgt[0])
    # src_unique = list(set(src.split()))
    # tgt_unique = list(set(tgt.split()))
    # src_unique = list(set(src))
    # tgt_unique = list(set(tgt))

for sdv, don in zip(sidn[:100000], wroibn[:100000]):
    src = sdv.strip()
    tgt = don.strip()
    # src = tokenizer.tokenize(sdv)
    # tgt = tokenizer.tokenize(don)
    src_unique = list(set(tokenizer2.tokenize(src)))
    tgt_unique = list(set(tokenizer2.tokenize(tgt.replace('</t>', '').replace('<t>', ''))))
    # src_unique = list(set(src.split()))
    # tgt_unique = list(set(tgt.split()))
    # src_unique = list(set(src))
    # tgt_unique = list(set(tgt))
    if len(tgt_unique) > 1:
        b = 0
        for w in tgt_unique:
            if w in src_unique:
                b += 1
        coverage = (b / len(tgt_unique))
        c.append(coverage)

print('subword  coverage:', sum(c) / len(c))
print()