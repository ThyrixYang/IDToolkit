import numpy as np

from .base import ParameterSpaceBase
from .category import CategorySpace

class CombineSpace(ParameterSpaceBase):
    
    def __init__(self, space_dict):
        self.space_dict = space_dict
        
        self.all_numerical = True
        for _, v in self.space_dict.items():
            if isinstance(v, CategorySpace):
                self.all_numerical = False
        
    def sample(self):
        res = {k: v.sample() for k, v in self.space_dict.items()}
        return res
    
    def check(self, param):
        for k, s in self.space_dict.items():
            assert k in param, f"{k} not in param"
            assert s.check(param[k]), f"{k}th param value {param[k]} not valid"
        return True
    
    def print_info(self):
        print(f"Help information of this combine parameter space")
        for k, s in self.space_dict.items():
            print(f"[{k}]: ", end=" ")
            s.print_info()
            
    def to_ray_space(self, continuous=False):
        ray_space = {k: s.to_ray_space(continuous) for k, s in self.space_dict.items()}
        return ray_space
    
    def to_zooopt_space(self):
        zooopt_space = {k: s.to_zooopt_space() for k, s in self.space_dict.items()}
        return zooopt_space
    
    def uniform_encode(self, param):
        encode_param = {}
        for k in self.space_dict.keys():
            if isinstance(self.space_dict[k], CategorySpace):
                encode_param[k] = self.space_dict[k].uniform_encode(param[k])
            else:
                encode_param[k] = param[k]
        return encode_param
    
    def uniform_decode(self, param):
        decode_param = {}
        for k in self.space_dict.keys():
            if isinstance(self.space_dict[k], CategorySpace):
                decode_param[k] = self.space_dict[k].uniform_decode(param[k])
            else:
                decode_param[k] = param[k]
        return decode_param
    
    def convert_param(self, param):
        converted_param = {}
        for k, v in param.items():
            converted_param[k] = self.space_dict[k].convert_param(v)
        return converted_param
    
    def to_numpy(self, param):
        np_params = []
        for k in self.space_dict.keys():
            np_params.append(self.space_dict[k].to_numpy(param[k]))
        np_params = np.concatenate(np_params, axis=0)
        return np_params
    
    def from_numpy(self, param):
        if not self.all_numerical:
            raise ValueError("Only support converting numerical parameters from numpy")
        param = np.reshape(param, (-1))
        assert param.shape[0] == len(self.space_dict)
        param_dict = {}
        for i, k in enumerate(self.space_dict.keys()):
            param_dict[k] = self.space_dict[k].clip(param[i])
        return param_dict