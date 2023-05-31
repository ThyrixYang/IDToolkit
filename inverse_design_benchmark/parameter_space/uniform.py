import numpy as np
import ray
try:
    from zoopt import ValueType
except:
    pass

from .base import ParameterSpaceBase

class UniformSpace(ParameterSpaceBase):
    
    def __init__(self, low, high):
        self.low = low
        self.high = high
        
    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)
    
    def check(self, param):
        return param >= self.low and param <= self.high
    
    def print_info(self):
        print(f"real value within [{self.low}, {self.high}), uniform sampling")
        
    def to_ray_space(self, continuous=True):
        return ray.tune.uniform(self.low, self.high)
    
    def to_zooopt_space(self):
        return [ValueType.CONTINUOUS, [self.low, self.high], 1e-15]
    
    def convert_param(self, param):
        try:
            param = float(param)
        except:
            raise ValueError(f"auto convert param failed with param={param}")
        return param
            
    def to_numpy(self, param):
        return np.array(param).reshape((1,))
    
    def clip(self, param):
        return np.clip(param, a_min=self.low, a_max=self.high)