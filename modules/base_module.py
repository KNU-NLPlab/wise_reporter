
class BaseModule():
    '''
    Base class of each module
    The abstract methods, shown below, should be implemented.
    '''
        
    def process_data(*params):
        '''
        
        '''
        raise NotImplementedError
    
    def get_viz_json(*params):
        '''
        
        '''
        raise NotImplementedError