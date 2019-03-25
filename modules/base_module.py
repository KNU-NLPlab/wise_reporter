
class BaseModule():
    '''
    Base class of each module
    The abstract methods, shown below, should be implemented.
    '''
    
    def __init__(*params):
        
        '''
        
        '''
        
        self.main_json = None
        self.detailed_json = None
        
        raise NotImplementedError
        
    def process_data(*params):
        '''
        
        '''
        raise NotImplementedError
    
    def generate_json(*params):
        '''
        
        '''
        raise NotImplementedError