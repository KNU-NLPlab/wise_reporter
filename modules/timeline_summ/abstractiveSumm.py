
m modules.timeline_summ.tools import etri_nlp
from modules.timeline_summ.tools import eventlist

from opennmt.onmt.translate.translator import make_translator

def get_morph_sentences(summary):
    morph_sentences, date_list = [], []
    for sentence in summary:
        normal_sentence = sentence['article'][0]['sentence']
        # make morph sentence
        etri = etri_nlp.Etri_nlp()
        parsed_json = etri.get_parsed_json(normal_sentence)
        morph_sentence = etri.make_morh_sentence(parsed_json)

        # appending morph_sentence
        morph_sentences.append(morph_sentence)
        date_list.append(sentence['date'])

    return morph_sentences, date_list

def get_event_tokens(morph_sentences):
    event_tokens = []
    event_list = eventlist.loadEventlist()

    for sentence in morph_sentences:
        morph_words = sentence.split(' ')
        event_token_list = []
        for word in morph_words:
            if word in event_list:
                event_token_list.append('1')
            else:
                event_token_list.append('0')
        event_tokens.append(' '.join(event_token_list))
    return event_tokens

def main(summary):

    morph_sentences, date_list = get_morph_sentences(summary)
    event_tokens = get_event_tokens(morph_sentences)



    return dates, summsentences
