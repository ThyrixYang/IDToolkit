
class ParameterSpaceBase:

    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError()

    def check(self, param):
        raise NotImplementedError()
    
    def print_info(self):
        raise NotImplementedError()

    def convert_param(self, param):
        raise NotImplementedError()