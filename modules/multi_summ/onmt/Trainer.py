from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import traceback

import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules

from torch.autograd import Variable

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0, reward=None):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()
        self.reward = reward
            

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        if self.reward is not None and stat.reward is not None:
            self.reward += stat.reward
#             print("trainer line:48 reward", self.reward)
        else:
            self.reward = stat.reward

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        return self.loss / self.n_words

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; xent: %6.2f; loss: %3.0f; " +
               "%3.0f; src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               self.loss,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        if self.reward is not None:
            print("reward : {}".format(self.reward / 50)) # default report every 
            self.reward = None
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, step):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper",  self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", lr, step)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, opt=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.progress_step = 0
        self.opt = opt
        self.reward = onmt.modules.Reward(opt.reward) #for reward

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
#         print("Trainer line:147 train_iter dataset")
#         print(train_iter.datasets)
        
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            # batch는 example과 유사하다고 생각하면 될 듯
            cur_dataset = train_iter.get_cur_dataset()
#             print("Trainer line:167 batch dataset fields src")
#             print(len(batch.dataset.fields['src'].vocab.freqs))
#             print(len(batch.dataset.fields['tgt'].vocab.freqs))
#             print(len(batch))
#             print("Trainer line:167 batch alignment")
#             print(batch.dataset) # textdataset
#             print(len(batch.dataset.examples)) # torchtext.examples
#             print(vars(batch.dataset.examples[0]))
#             print(len(batch.dataset.src_vocabs))
#             print(batch.dataset.src_vocabs[0].freqs) # 이게 dynamic dict에서 생성된 vocab
#             print(type(batch))
            
#             print(list(batch.fields)) # 이게 key형태로 저장되었음
# #             print(batch.indices)
#             print(batch.alignment)            
#             print()
#             print(batch.indices) # 10,000개식 저장됨
            __index = batch.indices.data[0]
#             print(batch.dataset.examples[__index].src)
#             input("Trainer line:184")
            if self.model.obj_f == "hybrid":
                self.train_loss.ml_loss_compute.cur_dataset = cur_dataset
                self.train_loss.rl_loss_compute.cur_dataset = cur_dataset
            else:
                self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            self.progress_step,
                            total_stats.start_time, self.optim.lr,
                            report_stats)
                    self.progress_step += 1

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text' or self.data_type == "hierarchical_text":
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            if self.model.model_type == "hierarchical_text":
                outputs, sent_attns, context_attns, dec_state = self.model(src, tgt, src_lengths, batch=batch)
                # Compute loss.
                batch_stats = self.valid_loss.monolithic_compute_loss(
                        batch, outputs, context_attns)                
            else:
                outputs, attns, _ = self.model(src, tgt, src_lengths, batch=batch)

                # Compute loss.
                batch_stats = self.valid_loss.monolithic_compute_loss(
                        batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_e%d_acc_%.2f_ppl_%.2f.pt'
                   % (opt.save_model, epoch, valid_stats.accuracy(),
                      valid_stats.ppl(), ))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
#             print("trainer line:306 new batch")
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            sec_dec_state = dec_state.clone() if dec_state else None
            if self.model.obj_f == "hybrid":
                sample_dec_state = dec_state.clone() if dec_state else None
                max_sample_dec_state = dec_state.clone() if dec_state else None
            
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text' or self.data_type == "hierarchical_text":
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt')
            
#             print("Trainer line 298: batch")
#             print(batch)
            
            # index만 출력함     
#             print("Trainer line 299: src")
#             print(src)
#             print("Trainer line 302: tgt")
#             print(tgt_outer)    
#             print("Trainer line 355: context mask", batch.context_mask)
#             print("Trainer line 355: indices", batch.indices)
#             print("Trainer line 355: batch", batch)
#             print("Trainer line 355: context length", batch.context_lengthes)
#             input()

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
                
                if self.model.obj_f == "ml":
                    #print("tariner line:373 model opt", self.model.model_type)
                    # 2. F-prop all but generator.
                    if self.grad_accum_count == 1:
                        self.model.zero_grad()
                        
                    if self.model.model_type == "hierarchical_text":
                        outputs, sent_attns, context_attns, dec_state = \
                            self.model(src, tgt, src_lengths, dec_state, batch)
    #                     print("Trainer line:346 outputs", outputs.size())                        
    #                     print("Trainer line:347 tgt", tgt.size())   
                        # 3. Compute loss in shards for memory efficiency.
                        batch_stats = self.train_loss.sharded_compute_loss(
                                batch, outputs, context_attns, j,
                                trunc_size, self.shard_size, normalization)
    #                     input("trainer line:377")

                    else:
                        outputs, attns, dec_state = \
                            self.model(src, tgt, src_lengths, dec_state, batch)
    #                     print("Trainer line:346 outputs", outputs.size())                        
    #                     print("Trainer line:347 tgt", tgt.size())   
                        # 3. Compute loss in shards for memory efficiency.
                        batch_stats = self.train_loss.sharded_compute_loss(
                                batch, outputs, attns, j,
                                trunc_size, self.shard_size, normalization)

                    # 4. Update the parameters and statistics.
                    if self.grad_accum_count == 1:
                        self.optim.step()
                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                    # If truncated, don't backprop fully.
                    if dec_state is not None:
                        dec_state.detach()
#                     input("trainer line 387")
                elif self.model.obj_f == "rl":
                    # 2. F-prop all but generator.
                    if self.grad_accum_count == 1:
                        self.model.zero_grad()
                        
#                     print("trainer line:394 copy", self.model.decoder._copy)
    
                    outputs, attns, dec_state, out_indices = \
                        self.model.sample(src, tgt, src_lengths, dec_state, batch, "sample")
                        
                   
                    _, _, sec_dec_state, max_out_indices = \
                        self.model.sample(src, tgt, src_lengths, sec_dec_state, batch, "greedy")                        
                      
                    batch_scores, sample_scores, max_scores, sample_alignments = self.reward.get_batch_reward(batch, out_indices, max_out_indices, self.model.decoder._copy)
#                     print("trainer line:371 batch_scores", batch_scores)
                    batch_scores = batch_scores.expand(out_indices.size(0),batch_scores.size(0))
#                     print("trainer line:398 batch_scores", batch_scores)
#                     print("trainer line:372 probs", torch.sum(probs,0))   
#                     loss = self.reward.criterion(probs, out_indices, Variable(batch_scores, requires_grad=False).cuda())
#                     loss = self.reward.criterion(max_probs, out_indices, Variable(batch_scores, requires_grad=False).cuda())
#                     print("Trainer line:377 outputs", outputs.size())
    
#                     print("Trainer line:378 tgt", tgt.size())
#                     print("Trainer line:379 out_indices", out_indices)
                    

#                     print("trainer line:384 batch.tgt",  batch.tgt)
#                     print("trainer line:384 batch.alignment",  batch.alignment)
#                     input()
#                     print("trainer line:384 batch.tht.le(3)",  batch.tgt[1:].le(3))
            
#                     print("trainer line:385 batch.tht.gt(3)",  batch.tgt[1:].gt(3))
#                     print("trainer line:386 padded sample 1", out_indices.mul(batch.tgt.data[1:].gt(3).long()))
#                     print("trainer line:386 padded sample 2", batch.tgt.data[1:].mul(batch.tgt.data[1:].le(3).long()))
                    # make sample to align with padding
#                     print("trainer line:405 eos", batch.dataset.fields['tgt'].vocab.stoi[onmt.io.EOS_WORD], onmt.io.EOS_WORD) # </s>
#                     print("trainer line:405 4", batch.dataset.fields['tgt'].vocab.itos[4], 4) # ,
#                     print("trainer line:405 5", batch.dataset.fields['tgt'].vocab.itos[5], 5) # 하        
#                     print("trainer line:405 6", batch.dataset.fields['tgt'].vocab.itos[6], 6) # 이                           
#                     print("trainer line:408 tgt",  tgt) # 51(max length?) * batchsize * 1
#                     print("trainer line:390 batch.tgt",  batch.tgt) # 51(max length) * batch size 
#                     print("trainer line:408 index",  out_indices) # 50(max length) * batch size
#                     print("trainer line:390 batch.tgt size",  batch.tgt)
#                     print("Trainer line:429 alignments", batch.alignment)
#                     batch.tgt.data[1:] = 1
#                     batch.tgt.data[1:out_indices.size(0)+1] =  out_indices
#                     print("Trainer line:429 batch.tgt.data", batch.tgt.data)
#                     print("Trainer line:430 batch.tgt.data[0]", out_indices)
                    batch.tgt.data = torch.cat((batch.tgt.data[0].unsqueeze(0), out_indices))
#                     print("trainer line:434 batch alignmnet", batch.alignment)    
                    batch.alignment = Variable(sample_alignments.contiguous(), requires_grad=False)
#                     print("trainer line:436 batch tgt", batch.tgt.size())    
#                     print("trainer line:436 after batch alignmnet", batch.alignment.size())    
#                     print("trainer line:437 output ", outputs.size())            

#                     print("trainer line:411 dec state",  dec_state)

#                     print("trainer line:431 batch.tgt size",  batch.tgt)                    
#                     print("Trainer line:432 sample alignments", sample_alignments)
#                     input()
                    
                   

                            # 3. Compute loss in shards for memory efficiency.
                    try:
                        batch_stats = self.train_loss.sharded_compute_loss(
                            batch, outputs, attns, j,
                            trunc_size, self.shard_size, normalization, rewards=Variable(batch_scores, requires_grad=False).cuda())

                    
    
#                     loss.backward()
#                     print("trainer line:372 loss", loss, j)   
#                     print("trainer line:385 batch.tgt", batch.tgt) # tgt_len * batch size
#                     input("trainer line:373")

                    
                    # 3. Compute loss in shards for memory efficiency.
#                     batch_stats = self.train_loss.sharded_compute_loss(
#                             batch, outputs, attns, j,
#                             trunc_size, self.shard_size, normalization, backward=False)
#                     batch_stats = Statistics(loss)

                    # 4. Update the parameters and statistics.
                        if self.grad_accum_count == 1:
                            self.optim.step()
                        total_stats.update(batch_stats)
                        report_stats.update(batch_stats)

                        # If truncated, don't backprop fully.
                        if dec_state is not None:
                            dec_state.detach()
                    except RuntimeError as e:
                        traceback.print_exc()
                        print("Trainer line:438 outputs", outputs.size())
                        print("Trainer line:439 batch alignment", batch.alignment.size()) 
                        sys.exit(1)
                elif self.model.obj_f == "hybrid":
                    # ml
                    # 2. F-prop all but generator.
                    if self.grad_accum_count == 1:
                        self.model.zero_grad()
                    outputs, attns, dec_state = \
                        self.model(src, tgt, src_lengths, dec_state, batch)

                    # rl
                    sample_outputs, sample_attns, sample_dec_state, out_indices = \
                        self.model.sample(src, tgt, src_lengths, sample_dec_state, batch, "sample")                    
               
                    _, _, max_sample_dec_state, max_out_indices = \
                        self.model.sample(src, tgt, src_lengths, max_sample_dec_state, batch, "greedy")                        
                      
                    batch_scores, sample_scores, max_scores, sample_alignments = self.reward.get_batch_reward(batch, out_indices, max_out_indices, self.model.decoder._copy)
                    batch_scores = batch_scores.expand(out_indices.size(0),batch_scores.size(0))

                    sample_batch_tgt = batch.tgt.clone()
                    sample_batch_tgt.data = torch.cat((batch.tgt.data[0].unsqueeze(0), out_indices))
                    sample_batch_alignment = Variable(sample_alignments.contiguous(), requires_grad=False)

                    # 3. Compute loss in shards for memory efficiency.
                    try:
                        batch_stats = self.train_loss.sharded_compute_loss(
                            batch, outputs, sample_outputs, attns, sample_attns, sample_batch_tgt, sample_batch_alignment, j,
                            trunc_size, self.shard_size, normalization, rewards=Variable(batch_scores, requires_grad=False).cuda())
                        

                    # 4. Update the parameters and statistics.
                        if self.grad_accum_count == 1:
                            self.optim.step()
                        total_stats.update(batch_stats)
                        report_stats.update(batch_stats)
                        
                       # If truncated, don't backprop fully.
                        if dec_state is not None:
                            dec_state.detach()                        

                        # If truncated, don't backprop fully.
                        if sample_dec_state is not None:
                            sample_dec_state.detach()
                        if max_sample_dec_state is not None:
                            max_sample_dec_state.detach()
                    except RuntimeError as e:
                        traceback.print_exc()
                        print("Trainer line:438 outputs", outputs.size())
                        print("Trainer line:439 batch alignment", batch.alignment.size()) 
                        sys.exit(1)
                torch.nn.utils.clip_grad_norm(self.model.parameters(),2)
                    
                    
        if self.grad_accum_count > 1:
            self.optim.step()
