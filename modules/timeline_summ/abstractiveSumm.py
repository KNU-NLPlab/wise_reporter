from __future__ import unicode_literals


def translate(sentences):
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args(args="--src 1 --model ./model/deletion_model.pt")

    ArgumentParser.validate_translate_opts(opt)

    translator = build_translator(opt, report_score=False)

    _, pred = translator.translate(
        src=sentences,
        batch_size=opt.batch_size,
        batch_type=opt.batch_type,
        attn_debug=opt.attn_debug
    )
    return pred

def get_morph_sentences(sentence_list):
    morph_sentences = []
    for sentence in sentence_list:
        morph_sentences.append(etri_nlp.get_morph_sentence(sentence))
    return morph_sentences


def binary_to_text(morph_sentences, pred_result):
    result_sentence = []
    for src, trg in zip(morph_sentences, pred_result):
        temp = []
        src = src.rstrip().split(' ')
        trg = trg.rstrip().split(' ')
        for i in range(len(src)):
            try:
                if trg[i] == '1':
                    temp.append(src[i])
            except:
                pass
        temp = ' '.join(temp)
        result_sentence.append(temp)
    return result_sentence


def main(summary):
    sentence_list = [sum['article'][0]['sentence'] for sum in summary]
    morph_sentences = get_morph_sentences(sentence_list)
    pred_result = translate(morph_sentences)
    pred_result = [result[0] for result in pred_result]
    translate_result = binary_to_text(morph_sentences, pred_result)

    for sum, result in zip(summary, translate_result):
        sum['article'][0]['sentence'] = result

    return summary



