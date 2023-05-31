import numpy as np
import ray
try:
    from zoopt import ValueType
except:
    pass

from .base import ParameterSpaceBase
from .uniform import UniformSpace

class CategorySpace(ParameterSpaceBase):
    
    def __init__(self, categories, p=None):
        self.categories = categories
        if p is not None:
            self.p = p
        else:
            self.p = np.ones(len(categories)) / len(categories)
        self.int_code = {c: float(i) for i, c in enumerate(self.categories)}
        assert len(self.categories) == len(self.p), "len(categories) != len(p)"
        assert np.isclose(np.sum(self.p), 1), f"sum(p) = {np.sum(self.p)} != 1"
        
    def sample(self):
        return np.random.choice(self.categories, p=self.p)
        
    def check(self, param):
        if isinstance(param, str):
            return param in self.categories
        else:
            if isinstance(param, float) or isinstance(param, int):
                return param >= 0 and param < len(self.categories)
            else:
                raise ValueError(f"Input Value Error in CategorySpace with value {param}")
    
    def print_info(self):
        print(f"category value within {self.categories}, default sampling distribution {self.p}")
        
    def to_ray_space(self, continuous):
        if continuous:
            return self.to_uniform_space().to_ray_space()
        else:
            return ray.tune.choice(self.categories)
        
    def to_zooopt_space(self):
        return (ValueType.GRID, self.categories)
    
    def uniform_decode(self, uniform_param):
        return self.categories[min(int(np.floor(uniform_param)), len(self.categories)-1)]
    
    def uniform_encode(self, param):
        return self.int_code[param]
    
    def to_uniform_space(self):
        return UniformSpace(low=0, high=len(self.categories))

    def convert_param(self, param):
        if str(param) in self.categories:
            return str(param)
        else:
            raise ValueError(f"auto convert param failed with param={param}")
        
    def to_numpy(self, param):
        int_value = int(self.int_code[param])
        np_param = np.zeros((len(self.categories),))
        np_param[int_value] = 1
        return np_param