import torch

import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from modules.timeline_summ.model.bert_morp_pytorch.src_tokenizer.tokenization_morp import BertTokenizer

class BertCompressor(nn.Module):
    def __init__(self, bert):
        super(BertCompressor, self).__init__()

        self.bert = bert
        self.fc_ext = nn.Linear(768 * 3, 1)

    def forward(self, inputs):
        batch_size, input_lengths = inputs.size()

        embed = self.bert.embeddings(inputs)

        attention_mask = (inputs != 0).float()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * model.config.num_hidden_layers

        encoded = model.encoder(embed, extended_attention_mask)[0]
        cls_encoded = model.pooler(encoded)  # CLS token

        context = torch.cat([cls_encoded.unsqueeze(dim=1).repeat(1, input_lengths - 2, 1),
                             encoded[:, 1:-1],
                             embed[:, 1:-1]], dim=-1)

        logits = torch.sigmoid(self.fc_ext(context))
        return logits.squeeze()

def sentence2idxs(sen, unk_idx=1, cls_idx=2, sep_idx=3):
    return [cls_idx] + [tokenizer.vocab[word] if word in tokenizer.vocab else unk_idx for word in sen.split()] + [sep_idx]


def resize_token_embeddings(model, new_num_tokens):
    old_embeddings = model.embeddings.word_embeddings

    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    if old_num_tokens == new_num_tokens:
        return

    # Build new embeddings
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device)

    # initialize all new embeddings (in particular added tokens)
    new_embeddings.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

    # Copy word embeddings from the previous weights
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

    model.embeddings.word_embeddings = new_embeddings
    return model


device = torch.device("cuda")

tokenizer = BertTokenizer('modules/timeline_summ/tools/vocab.korean_morp_added.list')
model = BertModel.from_pretrained('modules/timeline_summ/model/bert_morp_pytorch/').to(device)
resize_token_embeddings(model, len(tokenizer.vocab))



def main(summary):

    bertComp = BertCompressor(model).to(device)

    # load trained model
    # print('epoch 4')
    # snapshot = 'modules/timeline_summ/model/bert_morp_pytorch/timeline_summ_bertmodel.pt'
    # snapshot = 'modules/timeline_summ/model/bert_morp_pytorch/finetuning_6.pt'
    snapshot = 'modules/timeline_summ/model/bert_morp_pytorch/timeline_summ_bertmodel.pt'

    bertComp.load_state_dict(torch.load(snapshot, map_location = lambda storage, loc: storage))


    for timeline in summary:
        morph_tag = timeline['article'][1]['morph_tag']

        # prediction
        src = sentence2idxs(morph_tag)
        src = torch.tensor([src], device=device)
        prob = bertComp(src)
        label_list = (prob >= 0.5).tolist()
        label_list = [str(data) for data in label_list]
        sentence = [w[:w.index('/')] for w, d in zip(morph_tag.split(), label_list) if d == '1']
        timeline['compResult'] = sentence

    return summary