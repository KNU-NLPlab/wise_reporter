import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import onmt
import onmt.io
import onmt.modules
from onmt.Utils import aeq

import sys



# copied from CopyGeneratorCriterion
class RLGeneratorCriterion(object):
    def __init__(self, vocab_size, force_copy, pad, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.pad = pad

    def __call__(self, scores, align, target, rewards):
        # Compute unks in align and target for readability
        align_unk = align.eq(0).float()
        align_not_unk = align.ne(0).float()
        target_unk = target.eq(0).float()
        target_not_unk = target.ne(0).float()

        try:
        # Copy probability of tokens in source
            out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
#         out = scores.gather(1, align.view(-1, 1) + self.offset)
        except RuntimeError:
            print("RL line:34 socres size", scores.size())
            print("RL line:35 align size", align.size())
            sys.exit(1)
        
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            out = out + tmp.mul(target_not_unk)
            # Add score for when word is unk in both align and tgt
            out = out + tmp.mul(align_unk).mul(target_unk)
        else:
            # Forced copy. Add only probability for not-copied tokens
            out = out + tmp.mul(align_unk)
            
        # 05.18 remove copy related things
#         out = tmp
            
#         print("CopyGenerator line:136 out", out)
#         print("RL line:31 out",out.size())
#         print("RL line:32 scores",scores)
#         print("RL line:33 reward", rewards.size())
#         input("rl line:53")

        # Drop padding.
        # change sign 18.05.09
        loss = out.log().mul(target.ne(self.pad).float()) * rewards
        # change to orign 18.05.10
#         loss = -out.log().mul(target.ne(self.pad).float()) * rewards
#         print("CopyGenerator line:140 loss", loss)
        return loss


class RLGeneratorLossCompute(onmt.Loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """
    def __init__(self, generator, tgt_vocab,
                 force_copy, normalize_by_length, use_copy,
                 eps=1e-20, normalization="sents",
                 label_smoothing=0.0, initial_weight=None):
        super(RLGeneratorLossCompute, self).__init__(
            generator, tgt_vocab)

        # We lazily load datasets when there are more than one, so postpone
        # the setting of cur_dataset.
        self.cur_dataset = None
        self.force_copy = force_copy
        self.normalize_by_length = normalize_by_length
        self.use_copy = use_copy 
        if use_copy:
            self.criterion = RLGeneratorCriterion(len(tgt_vocab), force_copy,
                                                self.padding_idx)
            self.validate_loss_compute = onmt.modules.CopyGeneratorLossCompute(generator, tgt_vocab,
                 force_copy, normalize_by_length,
                 eps)            
        else:
            assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)
            if label_smoothing > 0:
                # When label smoothing is turned on,
                # KL-divergence between q_{smoothed ground truth prob.}(w)
                # and p_{prob. computed by model}(w) is minimized.
                # If label smoothing value is set to zero, the loss
                # is equivalent to NLLLoss or CrossEntropyLoss.
                # All non-true labels are uniformly set to low-confidence.
                self.criterion = nn.KLDivLoss(size_average=False)
                one_hot = torch.randn(1, len(tgt_vocab))
                one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
                one_hot[0][self.padding_idx] = 0
                self.register_buffer('one_hot', one_hot)
            else:
                if initial_weight is not None:
                    weight = torch.Tensor(initial_weight)
                    print("Loss line:186 Initialize weights with parameter {}".format(weight.size()))
                else:
                    weight = torch.ones(len(tgt_vocab))
                weight[self.padding_idx] = 0
                self.criterion = nn.NLLLoss(weight, size_average=False)
            self.confidence = 1.0 - label_smoothing
            self.validate_loss_compute = onmt.Loss.NMTLossCompute(
                generator, tgt_vocab,
                label_smoothing=0.0, initial_weight=None)
        
    
    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, backward=True, rewards=None):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        assert rewards is not None
        
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns, rewards)
        
        
        
#         print("RL line:131 range", range_)

        for shard in onmt.Loss.shards(shard_state, shard_size):
#             print("Loss, line:123", shard)
            if self.use_copy:
                if 'rewards' not in shard:
                    print("rl line:121 validate")
                    loss, stats = self.validate_loss_compute(batch, **shard)        
                else:
                    loss, stats = self._compute_loss(batch, **shard)
            else:
                if 'rewards' not in shard:
                    print("rl line:121 validate")
                    loss, stats = self.validate_loss_compute(batch, **shard)        
                else:
                    loss, stats = self._no_copy_compute_loss(batch, **shard)                
                
            if backward:
                loss.div(normalization).backward()
            batch_stats.update(stats)
#         input()
        return batch_stats        

    def _make_shard_state(self, batch, output, range_, attns, rewards=None):
        """ See base class for args description. """

#         print("CopyGenerator line 163",batch.src)
#         print("CopyGenerator line 164",batch.tgt[range_[0] + 1: range_[1]])
#         print("CopyGenerator line 165",batch.alignment[range_[0] + 1: range_[1]])
#         print("CopyGenerator line 166",type(batch))
#         print("RL line:141 rewards size", rewards.size())
#         print("RL line:141 batch.alignment size", batch.alignment.size())
#         print("RL line:141 tar size", batch.tgt)
#         print("RL line:141 range", range_)
#         print("RL line:159 output", output.size())
        if not self.use_copy:
            if rewards is None:
                return {
                    "output": output,
                    "target": batch.tgt[range_[0] + 1: range_[1]],
                }
            else:
                return {
                    "output": output,
                    "target": batch.tgt[range_[0] + 1: range_[1]],
                    "rewards": rewards # for shards
                }                
                
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")                

        if rewards is None:
            return {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "copy_attn": attns.get("copy"),
                "align": batch.alignment[range_[0] + 1: range_[1]],
            }        
    
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]],
            "rewards": rewards # for shards

        }
    
    def _no_copy_compute_loss(self, batch, output, target, rewards):
        scores = self.generator(self._bottle(output))
#         print("rl line:234 score size", scores.size()) # (target * batch )* vocablen
#         print("rl line:234 rewards size", rewards.size()) # target * batch
#         print("rl line:234 rewards size", target.size()) # target * batch
        
        
        stat_reward = (rewards.sum()/rewards.size()[1]).data[0]
        rewards = rewards.contiguous().view(-1)        
#         rewards = rewards.unsqueeze(1).expand_as(scores)
        
#         print("rl line:234 rewards size", rewards.size()) # target * batch
#         input("rl line:236")
                

        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)
        loss = self.criterion(scores * rewards.unsqueeze(1), gtruth)
#         torch.pow(scores, rewards)
#         print("rl line:253 loss", loss) # target * batch
#         print("rl line:253 loss", loss) # target * batch
#         input()
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        return loss, stats    

    def _compute_loss(self, batch, output, target, copy_attn, align, rewards):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
#         print("RL line:151 output", output.size())
#         print("RL line:151 align size", align.size())
#         print("RL line:177 reward size", rewards)
#         print("RL line:178 reward size", (rewards.sum()/rewards.size()[1]).data[0])
#         print("RL line:177 reward size", rewards.size())
#         input("RL line 177")
#         print("RL line:152 reward", rewards)

        stat_reward = (rewards.sum()/rewards.size()[1]).data[0]
        rewards = rewards.contiguous().view(-1)
        
#         rewards = rewards.expand_as(align).contiguous().view(-1)
#         print("RL line:153 reward expands", rewards.unsqueeze(1).expand_as(align))
        target = target.view(-1)
        align = align.view(-1)
#         print("CopyGenerator line:187", output)
        scores = self.generator(self._bottle(output),
                                self._bottle(copy_attn),
                                batch.src_map)
#         print("CopyGenerator line:191", scores)

        loss = self.criterion(scores, align, target, rewards)
        scores_data = scores.data.clone()
        scores_data = onmt.io.TextDataset.collapse_copy_scores(
                self._unbottle(scores_data, batch.batch_size),
                batch, self.tgt_vocab, self.cur_dataset.src_vocabs)
        scores_data = self._bottle(scores_data)
#         print("CopyGenerator line:202", scores_data)        
        
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.data.clone()
        correct_mask = target_data.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
        target_data = target_data + correct_copy

        # Compute sum of perplexities for stats
        loss_data = loss.sum().data.clone()
        stats = self._stats(loss_data, scores_data, target_data, reward=stat_reward)

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            pad_ix = batch.dataset.fields['tgt'].vocab.stoi[onmt.io.PAD_WORD]
            tgt_lens = batch.tgt.ne(pad_ix).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

#         print("RL line:207 loss summed?", loss)
#         input()
        return loss, stats

    