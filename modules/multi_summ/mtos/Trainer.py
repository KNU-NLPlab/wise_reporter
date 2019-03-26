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
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules

import random


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def xent(self):
        return self.loss / self.n_words

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
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; xent: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, epoch):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/ppl", self.ppl(), epoch)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), epoch)
        writer.add_scalar(prefix + "/tgtper",  self.n_words / t, epoch)
        writer.add_scalar(prefix + "/lr", lr, epoch)


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
                 norm_method="sents", grad_accum_count=1):
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

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()
        #self.denoise = onmt.modules.Denoising(0.1, 2) # noise probability, padding idx

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
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
            cur_dataset = train_iter.get_cur_dataset()
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
                            total_stats.start_time, self.optim.lr,
                            report_stats)

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
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None
                
            #src, src_lengths, noise_type = self.denoise.random_noise(src, src_lengths)

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            
            
            #outputs, attns, _ = self.model(src, tgt, src_lengths)
            outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths)
                

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
                      valid_stats.ppl() ))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None
                
            tgt_outer = onmt.io.make_features(batch, 'tgt')
            
            #print("trainer line:297, src, src_lengths", src, src_lengths)
            #src, src_lengths = self.denoise.noise_drop_swap(src, src_lengths)
            
            #prev_src_lengths = src_lengths.clone()
            #print("trainer line:308, src, src_lengths", src.squeeze(-1), src_lengths.view(1,-1))
            #src, src_lengths, noise_type = self.denoise.random_noise(src, src_lengths)
            #src, src_lengths, noise_type = self.denoise.seperate_noising_all(src, src_lengths)
            
            # 0: drop
            # 1 :swap
            # 2: replace
            # 3: drop swap
            # 4: replace swap
            # 5 :drop swap
            
           
                
            
            #print("trainer line:313, src, src_lengths", src.squeeze(-1), src_lengths.view(1,-1))
            #print("trainer line:314, noise_type", noise_type)
            
            # print(batch.dataset.fields['src'].vocab.stoi[onmt.io.NULL_WORD]) # idx: 2
            # print(batch.dataset.fields['src'].vocab.stoi[onmt.io.PAD_WORD]) # idx: 1
            #input("trainer line:316")
            
            for j in range(0, target_size-1, trunc_size):
                
                '''
                # denoising mechanism
                p = 0.1
                noise = []
                
                # drop
                for i in range(min(src_lengths)):                    
                    if min(src_lengths) - len(noise) < 2:
                        break
                    if random.random() <= 1-p:
                        noise.append(i)
                        
                # swap
                if len(noise) > 1:
                    for i in range(len(noise)-1):
                        if random.random() <= p:
                            tmp = noise[i]
                            noise[i] = noise[i+1]
                            noise[i+1] = tmp
                            i = i + 1
                        
                if len(noise) > 0:
                    # we assume training is processed on the gpu
                    noised_src = src.index_select(0, torch.autograd.Variable(torch.LongTensor(noise)).cuda() )
                    num_noised = src.size()[0]-len(noise)
                    
                    noised_src_lengths = src_lengths - (src.size()[0]-len(noise))
#                     print("## noised")
#                     print(noised_src_lengths)
#                     print(num_noised)
#                     print(noised_src_lengths[noised_src_lengths <= num_noised])
                    noised_src_lengths[noised_src_lengths < num_noised] = num_noised
#                     print(noised_src_lengths)
                else:
                    noised_src = src
                    noised_src_lengths = src_lengths
#                     print("## no noised")
                
#                 print(noised_src)
#                 print(src.size())
#                 print(noised_src_lengths)
#                 print(src_lengths)
                                                  
#                 print(type(noised_src))
#                 print(type(noised_src_lengths))
#                 print(type(src_lengths))
#                 input()   
                '''
                
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
                
                ###
#                 print(src)
#                 print(src_lengths)
#                 print(type(src))
#                 print(type(src_lengths))
#                 print(type(tgt))
#                 print(type(tgt_outer))
                #a = torch.index_select(src, 0, torch.LongTensor([1,2,3,4,5]))
                #print(a)
                #print(type(a))
#                 print(src.index_select(0, torch.autograd.Variable(torch.LongTensor([1,2,3,4,5]).cuda())))
#                 print(src_lengths - 1)
#                 print(src.size()[0])
#                 input()

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                try:
                    outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state, sort=True)
                except ValueError:
                    print("trainer:386, noise_type", noise_type)
                    print("trainer:387, prev_src_lengths", prev_src_lengths)
                    print("trainer:388, src_lengths", src_lengths)
                    input()
                    

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


        if self.grad_accum_count > 1:
            self.optim.step()
