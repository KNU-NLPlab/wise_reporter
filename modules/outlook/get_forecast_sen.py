import os
from modules.outlook.model import cnnmodel
from modules.outlook import prediction
from torchtext import data
import torch

class forecasting():
    def __init__(self, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
        self.model_name = 'modules/outlook/model/forecast_model.pt'

    def process_data(self, doc_info_list):
        '''
        Overrided method from BaseModule class
        Extract forecast sentences from articles
        Args:
            doc_info_list (list): a list of morph documents
        '''

        return self.forecast_sentences(doc_info_list)

    def forecast_sentences(self, doc_info_list):
        fields = torch.load('modules/outlook/model/fields.field')
        text_field = fields['text_filed']
        label_field = fields['label_field']
        kernel_sizes = '3,4,5'
        kernel_sizes = [int(k) for k in kernel_sizes.split(',')]

        cnn = cnnmodel.CNN_text(len(text_field.vocab), 64, len(label_field.vocab)-1, 100, kernel_sizes)
        cnn.load_state_dict(torch.load(self.model_name, map_location=lambda storage, loc: storage))

        forecast_list = []
        for doc in doc_info_list:
            ##test
            for sentence in doc['extract']:
                morph_sentence = [morp_info['lemma'] for morp_info in sentence['morph']]
                output = prediction.predict(morph_sentence, cnn, text_field, label_field)
                if output == '1':
                    forecast_list.append(sentence)
            #real one
            # for sentence in doc['anayzed_text']['sentence']:
            #     morph_sentence = [morp_info['lemma'] for morp_info in sentence['morp']]
            #     output = prediction.predict(morph_sentence)
        return forecast_list

    def make_morp_sentence(self, morph_json):
        morp_element_list = []
        for sentence in morph_json['sentence']:
            # get morp
            morp_elements = [morp_info['lemma'] for morp_info in sentence['morp']]
            morp_element_list.extend(morp_elements)
            # save to file
        return ' '.join(morp_element_list)
