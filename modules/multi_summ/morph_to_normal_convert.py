#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse
import codecs

from mtos.translate.Translator import make_translator

import mtos.io
import mtos.translate
import mtos
import mtos.ModelConstructor
import mtos.modules
import mtos.opts

import timeit

model_path = "modules/multi_summ/dataset_m2s2/1003_morph_to_src_syll_e10_acc_99.25_ppl_1.02.pt"
#model_path = "1003_morph_to_src_syll_e10_acc_99.25_ppl_1.02.pt"

res_path = "modules/multi_summ/dataset_m2s2/tmp_out_{}.txt"
input_path = "modules/multi_summ/dataset_m2s2/tmp_in_{}.txt"

opt = argparse.Namespace(alpha=0.0, attn_debug=False, batch_size=20, beam_size=5, beta=-0.04, block_ngram_repeat=0, coverage_penalty='none', data_type='text', dump_beam='', dynamic_dict=False, gpu=0, ignore_when_blocking=[], length_penalty='none', max_length=2000, max_sent_length=None, min_length=0, model=model_path, n_best=1, output=res_path.format(""), replace_unk=False, report_bleu=False, report_rouge=False, sample_rate=16000, share_vocab=False, src=input_path.format(""), src_dir='', stepwise_penalty=False, tgt=None, verbose=False, window='hamming', window_size=0.02, window_stride=0.01)
translator_m2n = make_translator(opt, report_score=True)

def to_syllable(sent):
    sent = sent.strip()
    sent = sent.replace('^', ' ')
    sent = sent.replace('  ', ' ')
    sent = sent.replace(' ', '^')
    sent = list(sent)
    sent = ' '.join(list(sent))

    return sent

def to_word(sent):
    sent = sent.strip()
    sent = sent.replace(' ', '')
    sent = sent.replace('^', ' ')

    return sent

def save_input_sent(file_path, sent_list):
    with open(file_path, 'w', encoding="utf-8") as out_file:
        for sent in sent_list:
            print(sent, file=out_file)

def convert(sent_list, keyword="default"):
    
    assert type(sent_list) == type(list())
    
    opt = argparse.Namespace(alpha=0.0, attn_debug=False, batch_size=20, beam_size=5, beta=-0.04, block_ngram_repeat=0, coverage_penalty='none', data_type='text', dump_beam='', dynamic_dict=False, gpu=0, ignore_when_blocking=[], length_penalty='none', max_length=2000, max_sent_length=None, min_length=0, model=model_path, n_best=1, output=res_path.format(keyword), replace_unk=False, report_bleu=False, report_rouge=False, sample_rate=16000, share_vocab=False, src=input_path.format(keyword), src_dir='', stepwise_penalty=False, tgt=None, verbose=False, window='hamming', window_size=0.02, window_stride=0.01)
    translator_m2n.out_file = codecs.open(opt.output, 'w', 'utf-8')
    
    syll_sent_list = ( to_syllable(sent) for sent in sent_list )
    save_input_sent(opt.src, syll_sent_list)

    translator = make_translator(opt, report_score=True)
    
    start = timeit.default_timer()
    converted_sent_list, _, _, _, _ = translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug)
    end = timeit.default_timer()
#    print("m to s takes {}s".format(end-start))
#    print("converted sent: ", converted_sent_list)
#    print("converted sent len: ", len(converted_sent_list))
    word_converted_sent_list = [to_word(converted_sent[0]) for converted_sent in converted_sent_list]
#    print("converted sent: ", word_converted_sent_list)
        
    # currently attns_info,oov_info only contain first index data of batch
    return word_converted_sent_list


def main(opt):
    translator = make_translator(opt, report_score=True)
    
    start = timeit.default_timer()
    _, attns_info, oov_info, copy_info = translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug)
    end = timeit.default_timer()
    print("Translation takes {}s".format(end-start))
    
    # currently attns_info,oov_info only contain first index data of batch
    return attns_info, oov_info, copy_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)

    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)
