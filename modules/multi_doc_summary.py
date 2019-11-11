import os

import multiprocessing as mp
from modules.multi_summ.multi_doc_sum import mds
from modules.base_module import BaseModule


class MultiDocSummary(BaseModule):
    '''
    The model for a summary using multiple documents
    Args:
        topic (string): a keyword query
    '''
    def __init__(self, topic, out_path, gpu="0"):
        '''
        Args:
            gpu (str or int): gpu number
        '''
        super(MultiDocSummary, self).__init__(topic, out_path)
        self.mds = mds(gpu)
        
    def process_data(self, documents_list):
        '''
        Overrided method from BaseModule class
        Generate a summary and convert results composed of morphemes to sentences with words
        Args:
            documents_list (list): a list of plain documents [str, str, ..., ]
        '''
        
        return self.mds.m_translate(documents_list)
        
    def generate_json(self):
        '''
        Overrided method from BaseModule class
        Return two json dictionary for visualization
        Args:
        '''
        return self.main_json, self.detail_json

if __name__ == '__main__':
    example = ['aa', 'bb', 'cc']
    print(multi_doc_sum.m_translate(example))
