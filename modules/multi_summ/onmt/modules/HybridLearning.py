import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import onmt
import onmt.io
import onmt.modules
from onmt.Utils import aeq

import sys


class HybridLossCompute(onmt.Loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """
    # apply_factor = 0.9984 # deep summ
    def __init__(self, generator, tgt_vocab,
                 force_copy, normalize_by_length,
                  apply_factor = 0.9984, eps=1e-20):
        super(HybridLossCompute, self).__init__(
            generator, tgt_vocab)

        # We lazily load datasets when there are more than one, so postpone
        # the setting of cur_dataset.
        self.ml_loss_compute = onmt.modules.CopyGeneratorLossCompute(
                generator, tgt_vocab, force_copy,
                normalize_by_length)
        self.rl_loss_compute = onmt.modules.RLGeneratorLossCompute(
                generator, tgt_vocab, force_copy,
                normalize_by_length)
        
        assert apply_factor >= 0. and apply_factor <= 1.
        self.apply_factor = apply_factor

    
    def sharded_compute_loss(self, batch, output, sample_outputs, attns, sample_attns, sample_batch_tgt, sample_batch_alignment,
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

        def shard_compute(batch, ml_shard_state, rl_shard_state, shard_size, batch_stats):

#         input()
            return loss_list, batch_stats          
        
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        ml_shard_state = self._make_shard_state(batch.tgt, batch.alignment, output, range_, attns)
        rl_shard_state = self._make_shard_state(sample_batch_tgt, sample_batch_alignment, sample_outputs, range_, sample_attns, rewards)
        
        shards = onmt.Loss.shards
        for ml_shard, rl_shard in zip(shards(ml_shard_state, shard_size), shards(rl_shard_state, shard_size)):
#             print("Loss, line:123", shard)
            ml_loss, stats = self.ml_loss_compute._compute_loss(batch, **ml_shard)        
            # be able to use same batch becuase subprocess doesn't use batch.tgt or batch.alignment
            rl_loss, stats = self.rl_loss_compute._compute_loss(batch, **rl_shard)
            if backward:
                loss = (1-self.apply_factor) * ml_loss.div(normalization) + self.apply_factor * rl_loss.div(normalization)
                loss.backward()
            batch_stats.update(stats)        
        
            
        return batch_stats
        
        
#         print("RL line:131 range", range_)
      


    def _make_shard_state(self, tgt, alignment, output, range_, attns, rewards=None):
        """ See base class for args description. """
#         if getattr(batch, "alignment", None) is None:
#             raise AssertionError("using -copy_attn you need to pass in "
#                                  "-dynamic_dict during preprocess stage.")
#         print("CopyGenerator line 163",batch.src)
#         print("CopyGenerator line 164",batch.tgt[range_[0] + 1: range_[1]])
#         print("CopyGenerator line 165",batch.alignment[range_[0] + 1: range_[1]])
#         print("CopyGenerator line 166",type(batch))
#         print("RL line:141 rewards size", rewards.size())
#         print("RL line:141 batch.alignment size", batch.alignment.size())
#         print("RL line:141 tar size", batch.tgt)
#         print("RL line:141 range", range_)
#         print("RL line:159 output", output.size())
        if rewards is None:
            return {
                "output": output,
                "target": tgt[range_[0] + 1: range_[1]],
                "copy_attn": attns.get("copy"),
                "align": alignment[range_[0] + 1: range_[1]],
            }        
    
        return {
            "output": output,
            "target": tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": alignment[range_[0] + 1: range_[1]],
            "rewards": rewards # for shards

        }
