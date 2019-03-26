import argparse
import torch
import codecs
import os
import math
import sys

from torch.autograd import Variable
from itertools import count

import mtos.ModelConstructor
import mtos.translate.Beam
import mtos.io
import mtos.opts

torch.backends.cudnn.enabled = False

def make_translator(opt, report_score=True, out_file=None):
#    if out_file is None:
#        out_file = codecs.open(opt.output, 'w', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    sys.argv = ["/workspace"]
    
    dummy_parser = argparse.ArgumentParser(description='train.py')
    mtos.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        mtos.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    scorer = mtos.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam",
                        "data_type", "replace_unk", "gpu", "verbose"]}

    translator = Translator(model, fields, global_scorer=scorer,
                            out_file=out_file, report_score=report_score,
                            copy_attn=model_opt.copy_attn, **kwargs)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`mtos.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """

    def __init__(self,
                 model,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 gpu=False,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 sample_rate='16000',
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 use_filter_pred=False,
                 data_type="text",
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None,
                 idf_attn_weight=False
                 
                ):
        self.gpu = gpu
        self.cuda = gpu > -1

        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.idf_attn_weight = idf_attn_weight
        
        # hardcoded to load idf value
        if self.idf_attn_weight:
            print("Translator line:127 Load idf value by file and revise num is 1, hard coded")
            idf_file_path = "idf_info.txt"
            
            
            src_vocab = self.fields["src"].vocab
            self.idf_attn_weight_list = [1] * len(src_vocab)
            
            with open(idf_file_path, 'r', encoding="utf-8") as idf_file:
                for line in idf_file:
                    word, freq, weight = line.strip().split('\t')
                    idx = src_vocab.stoi[word]
                    weight = float(weight)
                    if weight > 1 and freq != '0':
                        self.idf_attn_weight_list[idx] = weight
            if self.cuda:
                self.idf_attn_weights = torch.Tensor(self.idf_attn_weight_list).cuda()
            else:
                self.idf_attn_weights = torch.Tensor(self.idf_attn_weight_list).cuda()
#             print("Translator line:127 Complete load idf weights from file,len :  {} hard coded".format(len(self.idf_attn_weight_list)))
#             print("Translator line:142 idf tensor :  ", self.idf_attn_weights)
                
            

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self, src_dir, src_path, tgt_path,
                  batch_size, attn_debug=False):
        data = mtos.io.build_dataset(self.fields,
                                     self.data_type,
                                     src_path,
                                     tgt_path,
                                     src_dir=src_dir,
                                     sample_rate=self.sample_rate,
                                     window_size=self.window_size,
                                     window_stride=self.window_stride,
                                     window=self.window,
                                     use_filter_pred=self.use_filter_pred)

        data_iter = mtos.io.OrderedIterator(
            dataset=data, device=self.gpu,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = mtos.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        # for demo page
        attns_info = []
        oov_info = []
        copy_info = []
        n_pred_list = []

        for batch in data_iter:
            def check_oov(batch, vocab):
#                 print("Translator line:163 len batch", len(batch))
#                 print("Translator line:163 unk index", vocab.stoi["<unk>"]) # 0
#                 print("Translator line:163 2 token", vocab.itos[2])
                unk_index = vocab.stoi["<unk>"]
                batch_oov_indices = []
                
                for i in range(len(batch)):
                    length = batch.src[1][i]
                    oov_indices = [ 1 if idx == unk_index else 0 for idx in batch.src[0].data[:,i][:length]]
                    batch_oov_indices.append(oov_indices)
                
                return batch_oov_indices
                    
#                     print("Translator line:173 batch src", batch.src[0].data[:,i])
#                     print("Translator line:173 batch src", batch.src[1][i])
#                     print("Translator line:173 oov indices", oov_indices)
#                     print("Translator line:173 oov indices len", len(oov_indices))
#                     input()
            if len(batch) == 1: # assume demo page
                oov_info = check_oov(batch, self.fields["src"].vocab)

            #print("Translator line:216 batch", batch)
            batch_data = self.translate_batch(batch, data)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[0]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                #self.out_file.write('\n'.join(n_best_preds) + '\n')
                #self.out_file.flush()
                n_pred_list.append(n_best_preds)

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    os.write(1, output.encode('utf-8'))

                # Debug attention.
                #print("Translator line:182 attn", trans.attns[0])
                #print("Translator line:182 attn sum", torch.sum(trans.attns[0],0).view(1,-1))
#                 print("Translator line:182 copy", trans.copys[0]) # batch size * 1
#                 print("Translator line:183 trans.src_raw", len(trans.src_raw))

                # for demo page
                attns_info.append( torch.sum(trans.attns[0],0).tolist())
                if trans.copys is not None:                    
                    copy_info.append(trans.copys[0][0].squeeze(1).tolist())
#                 print(copy_info)
#                 print("Translator line:183 pred_sents[0]", trans.pred_sents[0])  
#                 print("Translator line:183 pred_sents[0] len", len(trans.pred_sents[0]))                    
                
#                 input("Translator line:213 stop")
                if attn_debug:
                    srcs = trans.src_raw
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *trans.src_raw) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    os.write(1, output.encode('utf-8'))
#                 input()
            batch = None

        if self.report_score:
            self._report_score('PRED', pred_score_total, pred_words_total)
            if tgt_path is not None:
                self._report_score('GOLD', gold_score_total, gold_words_total)
                if self.report_bleu:
                    self._report_bleu(tgt_path)
                if self.report_rouge:
                    self._report_rouge(tgt_path)

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return n_pred_list, all_scores, attns_info, oov_info, copy_info

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [mtos.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[mtos.io.PAD_WORD],
                                    eos=vocab.stoi[mtos.io.EOS_WORD],
                                    bos=vocab.stoi[mtos.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = mtos.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states)
        #dec_states_2 = self.model.decoder2.init_decoder_state(
        #    src, memory_bank, enc_states)
        
   
        
#         print("Translator line:338 src", src) # src_len * batch * 1
#         print("Translator line:339 memory_bank", memory_bank) # src len * batch * hidden
#         print("Translator line:339 src_lengths", src_lengths[0]) # src len * batch * hidden
        if self.idf_attn_weight and src_lengths[0] <= 2000:
          idf_size = self.idf_attn_weights.size(0)
#         print("Translator line:346 expand idf", self.idf_attn_weights.unsqueeze(0).repeat(src.size(0), 1).contiguous()) # src_len * idf size
#         print("Translator line:346 expand src squeeze -1", src.data.squeeze(-1)) # src_len * src_vocab size

#         print("Translator line:346 expand ooi", sum(src.data > idf_size)) # src_len * idf size       
#         print("Translator line:346 expand ooi", sum(src.data > idf_size+1)) # src_len * idf size       
          idf_attn_weights = None
          idf_attn_weights = torch.gather(self.idf_attn_weights.unsqueeze(0).expand(src.size(0), -1).contiguous(), 1, src.data.squeeze(-1).contiguous())
    
#         print("Translator line:339 idf attn weights", idf_attn_weights)
#         idf_attn_weights = rvar(idf_attn_weights)
          idf_attn_weights = idf_attn_weights.repeat(1, beam_size)
#         print("Translator line:339 idf attn weights", idf_attn_weights)
#         print("Translator line:339 idf", memory_bank)
        else:
          idf_attn_weights = None
        

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)        
        #dec_states_2.repeat_beam_size_times(beam_size)        
        
#         print("Translator line:338 src", src)
#         print("Translator line:355 memory_length", memory_lengths)
#         print("Translator line:355 src_lengths", src_lengths)
#         print("Translator line:355 memory_bank", memory_bank)
#         input()        
        
        #self.model.decoder.init_attn_history() # init attn history in decoder for new attention

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))
#             print("Translator line:295 inp", inp)
#             input("translator line296")

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)
#             print("Translator line:310 inp", inp)
#             input()

            # Run one step.
            # dec_out, dec_states, attn = self.model.decoder(
            #   inp, memory_bank, dec_states, memory_lengths=memory_lengths, idf_weights=idf_attn_weights)
            #dec_out = dec_out.squeeze(0)
            
            decoder_outputs, dec_states, attns = \
                self.model.decoder(inp, memory_bank,
                             dec_states,
                             memory_lengths=memory_lengths)

            #decoder_outputs_2, dec_states_2, attns_2, weight_lists_2 = \
            #    self.model.decoder2(inp, memory_bank,
            #                 dec_states_2,
            #                 memory_lengths=memory_lengths)               
            # dec_out: beam x rnn_size                
                
            #print("model line:603 enc_context", enc_context) # batch * 1 * hidden
            #print("model line:603 memory_bank", memory_bank) # src_len * batch * hidden
            #print("model line:603 weight_lists_1", weight_lists_1) # src_len * batch * 1
            #aggregated_weight = torch.cat([weight_lists_1, weight_lists_2], 2) # tgt_len * batch * 2
            #tgt_len, batch_l, _ = aggregated_weight.size()
            #aggregated_weight = self.model.sm(aggregated_weight.view(tgt_len * batch_l, 2)).view(tgt_len, batch_l, 2)
            #print("model line:603 aggregated_weight", aggregated_weight) # src_len * batch * 2
            #print("model line:603 aggregated_weight", aggregated_weight[:,:,0]) # src_len * batch * 2
            #print("model line:603 softmax") # 

            #print("model line:603 decoder_outputs_1", decoder_outputs_1[0])
            #decoder_outputs = decoder_outputs_1 * aggregated_weight[:,:,0].unsqueeze(2)        
            #decoder_outputs_2 = decoder_outputs_2 * aggregated_weight[:,:,1].unsqueeze(2)        
            #print("model line:603 decoder_outputs_1", decoder_outputs_1[0]) # tgt_len * batch * hideen
            #print("model line:603 dec_state_1", dec_state_1.size())

            #dec_out = self.model.linear_out(torch.cat([decoder_outputs_1, decoder_outputs_2], 2))                
            dec_out = decoder_outputs.squeeze(0)
            # dec_out: beam x rnn_size
            
            

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attns["std"])
            else:
                # assume demo page
                out, p_copy = self.model.generator.forward(dec_out,
                                                   attns["copy"].squeeze(0),
                                                   src_map, require_copy_p=True)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attns["copy"])
                beam_copy = unbottle(p_copy)
            # (c) Advance each beam.
#             print("Translator line:353 attn", beam_attn) # 
#             print("Translator line:353 attn copy", attn["copy"])
#             print("Translator line:353 p_copy", p_copy) # baem*batch
#             print("Translator line:353 unbottle p_copy", unbottle(p_copy)) # beam, batch, 1
#             print("Translator line:353 out", out)
#             input()
            for j, b in enumerate(beam):
                if not self.copy_attn:
                        b.advance(out[:, j],
                                  beam_attn.data[:, j, :memory_lengths[j]])
                        dec_states.beam_update(j, b.get_current_origin(), beam_size)
                        
                else:
                        b.advance(out[:, j],
                                  beam_attn.data[:, j, :memory_lengths[j]], copy_out=beam_copy.data[:,j,:])
                        dec_states.beam_update(j, b.get_current_origin(), beam_size)  
#                        dec_states_2.beam_update(j, b.get_current_origin(), beam_size)


        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
#         if "tgt" in batch.__dict__:
#             ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": [],
              "copy":[]}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn, copy = [], [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                if len(b.copy_p) != 0:
                    hyp, att, copy_p = b.get_hyp(times, k)
                    copy.append(copy_p)
                else:
                    hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
            if len(copy) != 0:
                ret["copy"].append(copy)  
        if len(copy) == 0:
            ret.pop("copy")
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = mtos.io.make_features(batch, 'src', data_type)
        tgt_in = mtos.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        
        self.model.decoder.init_attn_history() # init attn history in decoder for new attention        
        
        dec_out, _, _ = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[mtos.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores

    def _report_score(self, name, score_total, words_total):
        try:
            print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
            name, score_total / words_total,
            name, math.exp(-score_total / words_total)))
        except OverflowError:
            print("Overflow occured")
            print("Translator line 521 score_total", score_total)
            print("Translator line 521 words_total", words_total)

    def _report_bleu(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        print()

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
                                      % (path, tgt_path, self.output),
                                      stdin=self.out_file,
                                      shell=True).decode("utf-8")

        print(">> " + res.strip())

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=self.out_file).decode("utf-8")
        print(res.strip())
