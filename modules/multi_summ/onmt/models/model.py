""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch
import pickle
from pytorch_pretrained_bertt import BertModel, BertTokenizer, BertConfig

class Adapter(nn.Module):
    def __init__(self, dim, down_dim):
        super(Adapter, self).__init__()
        self.downstream = nn.Linear(dim, down_dim)
        self.activation = nn.Tanh()
        self.upstream = nn.Linear(down_dim, dim)

    def forward(self, x):
        _ = self.downstream(x)
        _ = self.activation(_)
        out = self.upstream(_)

        return out + x


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        # self.encoder = encoder
        # 기본 128
        down = 128
        # self.encoder = BertModel.from_pretrained('bert-base-uncased').to('cuda')
        self.encoder = BertModel(BertConfig.from_json_file('./bert_eojeol_pytorch/bert_config.json'))
        # self.encoder = BertModel.from_pretrained('./models/korean_bert')
        for i, _ in enumerate(self.encoder.parameters()):
            _.requires_grad = False
        # for _ in self.encoder.embeddings.parameters():
        #     _.requires_grad = False
        # for i, _ in enumerate(self.encoder.encoder.layer):
        #     if i < 6:
        #         for __ in _.parameters():
        #             __.requires_grad = False
        self.segment = pickle.load(open('./bert_eojeol_pytorch/segment.pkl', 'rb'))
        # self.encoder.encoder.adapter = nn.ModuleList([Adapter(768, 256).cuda() for _ in range(len(self.encoder.encoder.layer._modules))])
        for a in self.encoder.encoder.layer:
            a.attn_adapter = Adapter(768, down).cuda()
            a.ffn_adapter = Adapter(768, down).cuda()
        # for a in self.encoder.transformer:
        #     a.attn_adapter = Adapter(512, down).cuda()
        #     a.ffn_adapter = Adapter(512, down).cuda()
        self.decoder = decoder
        for i, _ in enumerate(self.decoder.parameters()):
            _.requires_grad = False
        for a in self.decoder.transformer_layers:
            a.attn_adapter = Adapter(768, down).cuda()
            a.context_adapter = Adapter(768, down).cuda()
            a.ffn_adapter = Adapter(768, down).cuda()
        # for a in self.decoder.transformer_layers:
        #     a.attn_adapter = Adapter(512, down).cuda()
        #     a.context_adapter = Adapter(512, down).cuda()
        #     a.ffn_adapter = Adapter(512, down).cuda()
        # print()
        # for i, _ in enumerate(self.decoder.transformer_layers):
        #     if i > 1:
        #         for __ in _.parameters():
        #             __.requires_grad = True
        # print()
        # self.param = nn.Parameter(torch.randn(self.encoder.total_hidden_dim * 4, dtype=torch.float32,
        #                                       device=torch.device('cuda')).view(1, -1, 1))

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            bptt (:obj:`Boolean`):
                a flag indicating if truncated bptt is set. If reset then
                init_state

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        # model.eval()
        # mask = (src[:, :, 0].transpose(0, 1).data.eq(0) ^ torch.ones_like(src.transpose(0, 1).squeeze(2),
        #                                                                   dtype=torch.uint8))
        # enc_state, memory_bank, _ = self.encoder(src.squeeze(2).transpose(0, 1), attention_mask=mask,
        #                                          output_all_encoded_layers=False, output_embeddings=True, adapter=True)

        aa = []
        for a in src.squeeze(2).transpose(0, 1):
            bb = []
            ma = True
            for b in a:
                if self.segment[int(b)]:
                    bb.append(ma)
                    ma = not ma
                else:
                    bb.append(ma)
            aa.append(bb)
        mask = (src[:, :, 0].transpose(0, 1).data.eq(0) ^ torch.ones_like(src.transpose(0, 1).squeeze(2),
                                                                          dtype=torch.uint8))
        enc_state, memory_bank, _ = self.encoder(src.squeeze(2).transpose(0, 1),
                                                 token_type_ids=torch.tensor(aa).type(torch.int64).to('cuda'),
                                                 attention_mask=mask, output_all_encoded_layers=False,
                                                 output_embeddings=True, adapter=True)


        # enc_state, memory_bank, _ = self.encoder(src.squeeze(2).transpose(0, 1), attention_mask=mask,
        #                                          output_all_encoded_layers=False, output_embeddings=True, adapter=False)

        # cls_bank = memory_bank[:, 0:1, :]
        memory_bank = torch.cat([memory_bank.transpose(0, 1)[:lengths[i], i:i + 1, :] for i in range(lengths.size(0))],
                                0)

        enc_state = torch.cat([enc_state.transpose(0, 1)[:lengths[i], i:i + 1, :] for i in range(lengths.size(0))], 0)
        #
        # enc_state, memory_bank, lengths = self.encoder(src, lengths, adapter=True)
        # memory_bank = torch.cat([memory_bank[:lengths[i], i:i + 1, :] for i in range(lengths.size(0))],
        #                         0)
        # enc_state = torch.cat([enc_state[:lengths[i], i:i + 1, :] for i in range(lengths.size(0))], 0)

        full_lengths = lengths.sum()
        if bptt is False:
            self.decoder.init_state(torch.cat([src[:lengths[i], i:i + 1, :] for i in range(lengths.size(0))], 0),
                                    memory_bank, enc_state)
        # dec_out, attns = self.decoder(tgt, memory_bank, memory_lengths=full_lengths, adapter=True, cls_bank=cls_bank)
        # dec_out, attns = self.decoder(tgt, memory_bank, memory_lengths=full_lengths, adapter=False)
        dec_out, attns = self.decoder(tgt, memory_bank, memory_lengths=full_lengths, adapter=True)

        return dec_out, attns, None

