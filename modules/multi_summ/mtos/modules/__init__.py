from mtos.modules.UtilClass import LayerNorm, Elementwise
from mtos.modules.Gate import context_gate_factory, ContextGate
from mtos.modules.GlobalAttention import GlobalAttention
from mtos.modules.ConvMultiStepAttention import ConvMultiStepAttention
from mtos.modules.ImageEncoder import ImageEncoder
from mtos.modules.AudioEncoder import AudioEncoder
from mtos.modules.CopyGenerator import CopyGenerator, CopyGeneratorLossCompute
from mtos.modules.StructuredAttention import MatrixTree
from mtos.modules.Transformer import \
   TransformerEncoder, TransformerDecoder, PositionwiseFeedForward
from mtos.modules.Conv2Conv import CNNEncoder, CNNDecoder
from mtos.modules.MultiHeadedAttn import MultiHeadedAttention
from mtos.modules.StackedRNN import StackedLSTM, StackedGRU
from mtos.modules.Embeddings import Embeddings, PositionalEncoding
from mtos.modules.WeightNorm import WeightNormConv2d
from mtos.modules.Denoising import Denoising


from mtos.Models import EncoderBase, MeanEncoder, StdRNNDecoder, \
    RNNDecoderBase, InputFeedRNNDecoder, RNNEncoder, NMTModel

from mtos.modules.SRU import check_sru_requirement
can_use_sru = check_sru_requirement()
if can_use_sru:
    from mtos.modules.SRU import SRU


# For flake8 compatibility.
__all__ = [EncoderBase, MeanEncoder, RNNDecoderBase, InputFeedRNNDecoder,
           RNNEncoder, NMTModel,
           StdRNNDecoder, ContextGate, GlobalAttention, ImageEncoder,
           PositionwiseFeedForward, PositionalEncoding,
           CopyGenerator, MultiHeadedAttention,
           LayerNorm,
           TransformerEncoder, TransformerDecoder, Embeddings, Elementwise,
           MatrixTree, WeightNormConv2d, ConvMultiStepAttention,
           CNNEncoder, CNNDecoder, StackedLSTM, StackedGRU,
           context_gate_factory, CopyGeneratorLossCompute, AudioEncoder]

if can_use_sru:
    __all__.extend([SRU, check_sru_requirement])
