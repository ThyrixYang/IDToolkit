import json
import os
import pathlib
import hashlib
import multiprocessing
from typing import Dict

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import pandas as pd
import ray
import yaml
import torch
import joblib

from xgboost import XGBRegressor
from pytorch_lightning import LightningModule
from ..algorithms.nn_models.mlp import MLP
from ..algorithms.nn_models.cnn import CNN

# nn based substitute models need extra process
#    inputs should be normalized by std and mean recorded
substitute_nn_name_to_model: Dict[str, LightningModule] = {
    'cnn': CNN,
    'mlp': MLP
}

class EnvBase:
    '''
        base class for all envs, implement the functions as you need
    '''
    def __init__(self, name, seed, save_model, substitute_model, ensemble):
        self.name = name
        self.seed = seed
        self.save_substitute_model = save_model
        self.substitute_model_name = substitute_model
        self.ensemble = ensemble
        self.substitude_models = []
        # empty args.substitute_model means numerical simulator
        if self.substitute_model_name:
            env_dir = f"checkpoint/{self.name}"
            self.substitute_model_root = pathlib.Path(f"{env_dir}/{self.substitute_model_name}").resolve()
            if not self.substitute_model_root.is_dir():
                raise RuntimeError("Missing substitute models! "
                                   "It is recommended to use xgboost for color_filter "
                                   "or use cnn for tpv and multi_layer as a substitute model.")
            
            # std_mean file, stores the std and mean for inputs params 
            #     saved when trainning forward prediction models
            #     used when solving inverse problems by iterative optimizers with nn-based model as a surrogate of the simulator
            std_mean_file = f"{env_dir}/std_mean.yaml"
            if os.path.exists(std_mean_file):
                with open(f"{env_dir}/std_mean.yaml", "r") as f:
                    self.std_mean = yaml.safe_load(f)
        
    def hash_param(self, param):
        obj = json.dumps(param, sort_keys=True)
        hash_key = int(hashlib.sha1(obj.encode("utf-8")).hexdigest(), 16)
        return hash_key

    # used in forward function
    def load_models(self):
        sub_model_name = self.substitute_model_name
        if sub_model_name and not self.substitude_models:
            # nn based substitute models
            if sub_model_name in substitute_nn_name_to_model:
                # every subpath is named by unique seed
                for subpath in self.substitute_model_root.iterdir():
                    # only the first model with this seed is used
                    v0 = subpath / "version_0"
                    hparam_path = v0 / "hparams.yaml"
                    ckpdir = v0 / "checkpoints"
                    ckpts = list(ckpdir.iterdir())
                    assert len(ckpts) == 1, "checkpoint should only have one model"

                    model = substitute_nn_name_to_model[sub_model_name].load_from_checkpoint(
                        ckpts[0], hparams_file=hparam_path)
                    model.freeze()
                    self.substitude_models.append(model)

                    # only the first seed is used, if flag "--ensemble" is not specified
                    if not self.ensemble:
                        break
            else:
                # load sklearn interface based models
                for subpath in self.substitute_model_root.iterdir():
                    if sub_model_name == "xgboost":
                        model = XGBRegressor()
                        model.load_model(subpath)
                    else:
                        model = joblib.load(subpath)
                    self.substitude_models.append(model)

                    if not self.ensemble:
                        break

    # give the value by substitute models
    def env_forward_by_models(self, params: np.ndarray):
        '''
            params: 1d numpy array
        '''
        if self.substitute_model_name in substitute_nn_name_to_model:
            params = (params - np.array(self.std_mean['std'])) / (np.array(self.std_mean['mean']) + 1e-8)
            params = torch.tensor(params).float()
        result = 0
        for model in self.substitude_models:
            if self.substitute_model_name in substitute_nn_name_to_model:
                result += model(params.unsqueeze(0))[0].numpy()
            else:
                result += model.predict([params])[0]
        result /= len(self.substitude_models)
        return result

    def forward(self, param):
        # Load substitute model before forward
        if self.substitute_model_name:
            self.load_models()

        if hasattr(self, "hash_key_ref"):
            hash_key = self.hash_param(param)
            hash_keys = ray.get(self.hash_key_ref)
            if hash_key in hash_keys:
                value_index = hash_keys.index(hash_key)
                dataset_values = ray.get(self.dataset_values_ref)

                # luckily, a param in dataset occurs
                print("cache hit")
                return dataset_values[value_index]
            else:
                return self.env_forward(param)
        else:
            return self.env_forward(param)

    # get value by env
    #    even with substitute models, we need numerical simulator to evaludate the solution finally
    #    so option force_numerical exists
    def env_forward(self, param, force_numerical=False):
        raise NotImplementedError()
    
    def env_forward_with_order(self, inputs):
        param, order = inputs
        value = self.env_forward(param, force_numerical=True)
        return (value, order)

    def score(self, value):
        raise NotImplementedError()

    def visualize(self, param, save_path):
        raise NotImplementedError()

    def init_dataset(self, params, values):
        if (params is None) or (len(params) == 0):
            return
        values = np.stack(values)
        params = [self.parameter_space.convert_param(p) for p in params]
        self.hash_key_ref = ray.put([self.hash_param(p) for p in params])
        self.dataset_values_ref = ray.put(values)
    
    def batch_forward(self, params, num_process=-1):
        if num_process == -1:
            num_process = multiprocessing.cpu_count()
        ordered_inputs = [(p, o) for p, o in zip(params, range(len(params)))]
        with multiprocessing.Pool(processes=num_process) as pool:
            results = list(tqdm(
                pool.imap_unordered(self.env_forward_with_order, ordered_inputs), 
                total=len(params)))
        results = sorted(results, key=lambda x: x[1])
        values = [x[0] for x in results]
        return values
            
    def generate_dataset(self, 
                         num_data, 
                         num_process=-1,
                         seed=None,
                         save_dir=None):
        np.random.seed(seed)
        print("generating parameters...")
        params = [self.sample() for _ in range(num_data)]
        print("calculating simulation results...")
        values = self.batch_forward(params, num_process=num_process)
        if save_dir is not None:
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True) 
            params_df = pd.DataFrame(params)
            values_df = pd.DataFrame(values)
            hash_id = pd.util.hash_pandas_object(params_df, 
                                                 index=True, 
                                                 hash_key='0123456789123456',
                                                 categorize=True).values
            hash_id = hashlib.sha1(hash_id).hexdigest()
            params_df.to_csv(pathlib.Path(save_dir) / f"params_{hash_id}.csv")
            values_df.to_csv(pathlib.Path(save_dir) / f"values_{hash_id}.csv")
        return params, values

    @property
    def parameter_space(self):
        raise NotImplementedError()

    def sample(self):
        return self.parameter_space.sample()
    
    @property
    def get_input_dim(self):
        raise NotImplementedError()
    
    @property
    def get_output_dim(self):
        raise NotImplementedError()
    
    def dataset_to_numpy(self, params, values=None):
        params_np = np.stack([self.parameter_space.to_numpy(p) for p in params])
        if values is not None:
            values_np = np.stack(values)
            return params_np, values_np
        else:
            return params_np
    