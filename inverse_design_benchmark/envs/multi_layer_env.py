import numpy as np

from .base import EnvBase
from ..parameter_space.combine import CombineSpace
from ..parameter_space.category import CategorySpace
from ..parameter_space.uniform import UniformSpace

from .multi_layer_model import layer_thickness_range, layer_material_range, max_layer_num
from .multi_layer_model import simulate, load_target

class MultiLayerEnv(EnvBase):
    
    def __init__(self, seed=0, save_model=False, substitute_model_name="", ensemble=False):
        super().__init__("multi_layer", seed, save_model, substitute_model_name, ensemble)
        self.target, self.wavelength = load_target()
    
    def env_forward(self, param, force_numerical=False):
        self.parameter_space.check(param)

        if not force_numerical and self.substitute_model_name:
            result = self.env_forward_by_models(self.parameter_space.to_numpy(param))
        else:
            layer_material = []
            layer_thickness = []
            for i in range(max_layer_num):
                layer_material.append(param[f"layer_material_{i}"])
                layer_thickness.append(param[f"layer_thickness_{i}"])
            result = simulate(layer_material=layer_material,
                            layer_thickness=layer_thickness)
            
        return result
    
    def score(self, value):
        _score = -np.mean((value - self.target)**2)
        return _score
    
    @property
    def parameter_space(self):
        if not hasattr(self, "_parameter_space"):
            spaces = {}
            for ln in range(max_layer_num):
                spaces[f"layer_material_{ln}"] = CategorySpace(categories=layer_material_range)
                spaces[f"layer_thickness_{ln}"] = UniformSpace(
                    low=layer_thickness_range[0], 
                                        high=layer_thickness_range[1])
            self._parameter_space = CombineSpace(space_dict=spaces)
        return self._parameter_space
    
    @property
    def get_input_dim(self):
        return 7*10 + 10
    
    @property
    def get_output_dim(self):
        return 2001