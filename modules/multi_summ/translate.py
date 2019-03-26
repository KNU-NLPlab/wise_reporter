#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse

from onmt.translate.Translator import make_translator

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import onmt.opts

import timeit

# 2018/11/22
# GPU memory flush version
def sub_main(queue, opt):
    
    translator = make_translator(opt, report_score=True)
    
    start = timeit.default_timer()
    # ocntext attns info랑 raw attns info를 혼용중 나중에 꼭 수정해야 함
    _, attns_info, oov_info, copy_info, context_attns_info = translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug, raw_attn=True)
    end = timeit.default_timer()
    print("Translation takes {}s".format(end-start))
    
    # currently attns_info,oov_info only contain first index data of batch
    if len(context_attns_info) == 0:
        queue.put((attns_info, oov_info, copy_info))
        return attns_info, oov_info, copy_info
    else:
        queue.put((attns_info, oov_info, copy_info, context_attns_info))
        return attns_info, oov_info, copy_info, context_attns_info


def main(opt):
    translator = make_translator(opt, report_score=True)
    
    start = timeit.default_timer()
    # ocntext attns info랑 raw attns info를 혼용중 나중에 꼭 수정해야 함
    _, attns_info, oov_info, copy_info, context_attns_info = translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug, raw_attn=True)
    end = timeit.default_timer()
    print("Translation takes {}s".format(end-start))
    
    # currently attns_info,oov_info only contain first index data of batch
    if len(context_attns_info) == 0:
        return attns_info, oov_info, copy_info
    else:
        return attns_info, oov_info, copy_info, context_attns_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)

    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)
