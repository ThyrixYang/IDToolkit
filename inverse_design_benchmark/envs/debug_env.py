import numpy as np
import time
import random

from inverse_design_benchmark.parameter_space import CombineSpace, CategorySpace, UniformSpace

from .base import EnvBase

class DebugEnv(EnvBase):
    
    def __init__(self):
        super().__init__()
        self.target = np.zeros(2,)
        self.name = "debug_env"
    
    def env_forward(self, param):
        self.parameter_space.check(param)
        time.sleep(random.uniform(0.01, 0.2))
        p = np.array([param["x"], param["y"]]) \
            + np.array([float(param["z"][1:]), float(param["a"][1:])])
        return p
    
    def score(self, value):
        _score = -np.mean((value - self.target)**2)
        return _score
    
    @property
    def parameter_space(self):
        if not hasattr(self, "_parameter_space"):
            spaces = {
                "x": UniformSpace(low=-5, high=5),
                "y": UniformSpace(low=-5, high=5),
                "z": CategorySpace(categories=["_1", "_-1", "_3", "_-3", "_5", "_-5"]),
                "a": CategorySpace(categories=["_2", "_-2", "_4", "_-4", "_6", "_-6"])
            }
            self._parameter_space = CombineSpace(space_dict=spaces)
        return self._parameter_space
    
    @property
    def get_input_dim(self):
        return 14
    
    @property
    def get_output_dim(self):
        return 2
    
class DebugNumEnv(EnvBase):
    
    def __init__(self):
        super().__init__()
        self.target = np.zeros(2,)
        self.name = "debug_numerical_env"
    
    def env_forward(self, param):
        self.parameter_space.check(param)
        time.sleep(random.uniform(0.01, 0.2))
        p = np.array([param["x"], param["y"]])
        return p
    
    def score(self, value):
        _score = -np.mean((value - self.target)**2)
        return _score
    
    @property
    def parameter_space(self):
        if not hasattr(self, "_parameter_space"):
            spaces = {
                "x": UniformSpace(low=-5, high=5),
                "y": UniformSpace(low=-5, high=5),
            }
            self._parameter_space = CombineSpace(space_dict=spaces)
        return self._parameter_space
    
    @property
    def get_input_dim(self):
        return 2
    
    @property
    def get_output_dim(self):
        return 2