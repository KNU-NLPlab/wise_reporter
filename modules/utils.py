
def make_morp_sentence_list(etri_dict):
    return [' '.join([morp['lemma'] for morp in sentence['morp']]) for sentence in etri_dict['sentence']]