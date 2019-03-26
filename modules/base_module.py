
class BaseModule():
    '''
    Base class of each module
    The abstract methods, shown below, should be implemented.
    '''
        
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