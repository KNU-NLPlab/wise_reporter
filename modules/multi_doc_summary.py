import os

from modules.base_module import BaseModule

from modules.multi_summ.main import stc_extract, make_summary


class MultiDocSummary(BaseModule):
    '''
    The model for a summary using multiple documents
    Args:
        topic (string): a keyword query
    '''
    def __init__(self, topic):
        self.topic = topic

    def process_data(self, top_doc_morph_sentence_list, top_keyword_list):
        '''
        Overrided method from BaseModule class
        Generate a summary and convert results composed of morphemes to sentences with words
        Args:
            top_doc_morph_sentence_list (list): a list of morph sentences in top documents
            top_keyword_list (list): a list of keywords of top documents
        '''
        data_list = stc_extract(self.topic, top_keyword_list, top_doc_morph_sentence_list) #파라메터 추가
        
        print("paraent pid", os.getpid())
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        p = ctx.Process(target=make_summary, args=(queue, keyword, data_list))
        p.start()
        
        self.main_json, self.detail_json = queue.get()
    
        #p.join(3) # timeout 안 설정하면 안끝남
        
    def generate_json(self):
        '''
        Overrided method from BaseModule class
        Return two json dictionary for visualization
        Args:
        '''
        return self.main_josn, self.detail_json