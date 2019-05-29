
class BaseModule():
    '''
    Base class of each module
    The abstract methods, shown below, should be implemented.
    '''
    def __init__(self, topic, out_path):
        '''
        Args:
            topic (string): a keyword query
            out_path (string): a output path to save json files
        '''
        self.topic = topic
        self.out_path = out_path        
        
    def process_data(*params):
        '''
        Process input data and make it until just before visualization.
        '''
        raise NotImplementedError
    
    def get_viz_json(*params):
        '''
        Return tuple of two json for visualization as follows: (Main json, Detail json)
        '''
        raise NotImplementedError