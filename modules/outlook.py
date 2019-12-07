from modules.forecast.get_forecast_sen import forecasting
from modules.base_module import BaseModule

class ForecastSentence(BaseModule):
    '''
    The model for extraction of forecast sentence in articles
    Args:
        doc_list(list): article list to extract forecast sentences
    '''
    def __init__(self, topic, out_path):
        '''
        :param gpu: gpu number
        '''
        self.topic = topic
        self.out_path = out_path
        self.gpu = 0
        self.forecasting = forecasting()
        # super(ForecastSentence, self).__init__(topic, out_path)
        # self.forecasting = forecasting(gpu)

    def process_data(self, doc_list, number_of_sentences):
        '''
        Overrided method from BaseModule class
        Args:
            documents_list (list): a list of plain documents [str, str, ..., ]
        '''

        return self.forecasting.forecast_sentences(doc_list, number_of_sentences)
