# Inverse design benchmark
## Requirements

- rich==13.3.1
- pandas==1.5.3
- tqdm==4.64.1
- PyYAML==6.0
- ray[tune]==2.3.0
- pytorch_lightning==1.9.3
- joblib==1.2.0
- nevergrad==0.6.0
- xgboost==1.7.4
- nltk==3.8.1
- networkx==3.1



## Datasets

Datasets are saved at [google drive](https://drive.google.com/drive/folders/1uqiEhybzZ0TAY_ibnjE8XYX5IWG4zphC?usp=sharing) and [baidu netdisk](https://pan.baidu.com/s/15olbh_sxnolKtxUIA0wzmw?pwd=422k), you can directly unzip `datasets.zip` and use it.



## Usage


All the algorithms and problems implemented according to our API can run experiments with our utilities easily:

```bash
python experiments/main.py \
--method method_name \
--method_config method_config_path \
--env env_name \
--dataset_path dataset_path_for_training \ 
--eval_method eval_method \
--pred_num number_of_evaluation_budget \ 
--seeds random_seeds \
--log_path path_to_results \ 
--alg_args \
--save_substitute_model \
--substitute_model substitute_model_name \
--ensemble \
--evaluate_after_search
# --method_config support yaml format
# --eval_method support IID target, real target and forward prediction.
# alg_args overwrite some hyper-parameters in method_config, useful for tuning specific hyper-parameters.

# All the evaluation results with different configurations and random seeds are saved in log_path.
# And the users can use the functions we provided in experiments/plots.py to plot all the figures in the paper.
```



We provide several sample scripts in a folder named `sample_scripts`, and you can run experiments directly by running the following scripts.

```bash
# train two forward prediction models, both are saved for future use
bash sample_scripts/forward_prediction.sh color_filter cnn xgboosts

# train and evaluate deep inverse model
bash sample_scripts/inverse_design_deep.sh color_filter iid_target cvae

# iterative optimizer based on simulator
bash sample_scripts/inverse_desigh_iterative.sh color_filter iid_target random_search

# iterative optimzer based on substitute models and evaluated on simulator
bash sample_scripts/inverse_design_iterative_with_surrogate.sh \
	color_filter iid_target random_search cnn
```



## Algorithms and envs supported

### Algorithms

| Forward prediction                    | Inverse design(Iterative method)                    | Inverse design(Deep method)                        |
| ------------------------------------- | --------------------------------------------------- | -------------------------------------------------- |
| Linear Regression (LR)                | Random search (RS)                                  | Inverse Model (IM)                                 |
| Decision Tree (DT)                    | Sequential randomized coordinate shrinking (SRACOS) | Gradient Descent (GD)                              |
| Gradient-Boosted Decision Tree (GBDT) | Bayesian Optimization (BO)                          | Tandem                                             |
| Multilayer perceptron (MLP)           | Tree-structured Parzen Estimator Approach (TPE)     | Conditional Generative Adversarial Networks (CGAN) |
| Convolutional neural networks (CNNs)  | Evolution Strategy (ES)                             | Conditional Variational Auto-Encoder (CVAE)        |



### Envs

- Multi-layer Optical Thin Films (MOTFs)
- Thermophotovoltaics (TPV)
- Structural color filter (SCF)



## Core APIs

We take MOTFs problem as an example, while the other two problems are similar:

```python
from inverse_design_benchmark.envs import MultiLayerEnv
from inverse_design_benchmark.algorithms import NeuralOptAlgorithm, ZOOptAlgorithm
# The API design of deep learning methods and iterative methods are different.
# We choose VAE and SRACOS as example.

# Instance evaluation function
env = MultiLayerEnv()

# Load dataset
data_path = "./datasets/multi_layer"
data_params, data_values = load_dataset(data_path)

# Training and predicting with deep learning method, VAE as example
alg = NeuralOptAlgorithm(env=env, config={"net": "vae", **network_configuration})
alg.fit(data_params, data_values)
# Get the predicted parameters and corresponding scores.
predicted_design_parameters = alg.search(num_samples=pred_num)
scores = [env.score(v) for v in predicted_design_parameters]

# Iterative method, SRACOS as example
alg = ZOOptAlgorithm(env=env, config={"parallel_num": 32})
results = alg.fit_and_search( 
    num_pred=pred_num,
    dataset_parameters=data_params)
# results include all the tried design parameters and their scores for further analysis
```



To use substitute models: 

```python
# Instance evaluation function implemented by substitute model
env_substitute = MultiLayerEnv(substitute_model_name='cnn', ensemble=True)

# Load dataset
data_path = "./datasets/multi_layer"
data_params, data_values = load_dataset(data_path)

# Iterative method, SRACOS as example
alg = ZOOptAlgorithm(env=env_substitute, config={"parallel_num": 32})
intermediate_results = alg.fit_and_search( 
    num_pred=pred_num,
    dataset_parameters=data_params)

# Re-evaluate the intermediate results by numerical simulation.
pred_params = intermediate_results["pred_params"]
pred_values = env_substitute.batch_forward(pred_params)
pred_scores = [env_substitute.score(v) for v in pred_values]
```

We provide some examples of substitute models (five GBDT checkpoints for SCF, five CNN checkpoints for MOTFs and TPV) and save them at [google drive](https://drive.google.com/drive/folders/1uqiEhybzZ0TAY_ibnjE8XYX5IWG4zphC?usp=sharing) and [baidu netdisk](https://pan.baidu.com/s/15olbh_sxnolKtxUIA0wzmw?pwd=422k). You can directly unzip `checkpoints.zip`  to the directory `IDToolkit/` and use it, or you can train your own forward prediction model as a substitute model by running `sample_scripts/forward_prediction.sh`.



## Add new components

### Add algorithms

**Iterative method**

We take a particle swarm optimization (PSO) algorithm as an example to add a iterative method.

```python
import ray
from ray import tune
from ray.tune.search.nevergrad import NevergradSearch
import nevergrad as ng
from ray.air import RunConfig
import pathlib

from .opt_base import Algorithm, parse_ray_tune_results

class PSOAlgorithm(Algorithm):

    def __init__(self,
                 env,
                 config):
        super().__init__(env=env, config=config)

        
    def fit_and_search(self,
                       num_pred=1,
                       dataset_parameters=None,
                       seed=0):
        '''
        	num_pred: 
        		The number of solutions to be generated.
        	dataset_parameters: 
        		Train data
        '''
        if dataset_parameters is not None:
            num_samples = len(dataset_parameters) + num_pred
        else:
            num_samples = num_pred

        self.alg = NevergradSearch(
            points_to_evaluate=dataset_parameters,
            optimizer=ng.optimizers.PSO,
        )

        tuner = tune.Tuner(
            self.score_fn,
            tune_config=tune.TuneConfig(
                search_alg=self.alg,
                metric="score",
                num_samples=num_samples,
                mode="max",
            ),
            param_space=self.env.parameter_space.to_ray_space(),
            run_config=RunConfig(verbose=1, local_dir=self.ray_result_path),
        )
        ray_results = tuner.fit()
        results = parse_ray_tune_results(ray_results, num_samples=num_pred)
        return results
```



**Deep method**

We take a conditional invertible neural network (cINN) model as an example to add a deep method.

```python
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
# The FrEIA package is required for implementing cINN

def get_cINN_model(x_dim, y_dim, hidden_size, layer_num): 
    def subnet_fc(in_dim, out_dim): 
        return nn.Sequential(nn.Linear(in_dim, hidden_size), 
                             nn.ReLU(), 
                             nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                             nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                             nn.Linear(hidden_size, out_dim)) 
        cond_node = ConditionNode(y_dim) 
        nodes = [InputNode(x_dim, name='input')] 
        for i in range(layer_num): 
            nodes.append(Node(nodes[-1], GLOWCouplingBlock, 
                              {'subnet_constructor': subnet_fc, 'clamp': 2.0}, 
                              conditions=cond_node, 
                              name='coupling_{}'.format(i))) 
            nodes.append(Node(nodes[-1], PermuteRandom, 
                              {'seed': i}, name='permute_{}'.format(i))) 
        nodes.append(OutputNode(nodes[-1], name='output')) 
        nodes.append(cond_node) 
        return ReversibleGraphNet(nodes, verbose=False)
    
class cINN(LightningModule): 
    
    def __init__(self, 
                 x_dim, # The dimension of design parameters 
                 y_dim, # The dimension of design targets 
                 config): # Model config 
        super().__init__() 
        self.config = config 
        self.model = get_cINN_model(x_dim=x_dim, y_dim=y_dim, 
                                    hidden_size=config.hidden_size, 
                                    layer_num=config.layer_num) 
    def _shared_step(self, x, y): 
        z, log_jac_det = model( 
            x, c=y) 
        loss = torch.sum(z**2, dim=1, keepdim=True) * \ 
        	0.5 - log_jac_det.view((-1, 1)) 
        return loss 
    
    def training_step(self, batch, batch_index): 
        x, y = batch 
        loss = self._shared_step(x, y) 
        return loss 
    
    def validation_step(self, batch, batch_index): 
        x, y = batch 
        loss = self._shared_step(x, y) 
        self.log("val_loss", loss) 
        return loss 
    
    def predict_step(self, batch, batch_index): 
        y = batch[0] 
        batch_size = y.shape[0]
        z_sample = torch.randn(size=(batch_size, self.config.x_dim), 
                               device=y.device).float() 
        pred_x, _ = model(z_sample, c=y, rev=True) 
        return pred_x
        
```



### Add envs

We give a code example for adapting a Therapeutics Data Commons (TDC) problem into our IDToolkit here. 

```python
'''
Docking is a theoretical evaluation of affinity (free energy change of the binding process) between a ligand (a small molecule) and a target (a protein involved in a disease pathway). A docking evaluation usually includes conformational sampling of ligand and free energy change calculation. A molecule with higher affinity usually has a higher potential to poses higher bioactivity.
'''

import numpy as npfrom .base 
import EnvBasefrom ..parameter_space.combine 
import CombineSpacefrom ..parameter_space.category 
import CategorySpacefrom ..parameter_space.uniform 
import UniformSpace

# We need to install PyTDC at first https://tdcommons.ai/
# The Oracle function is used to evaluate the bioactivity of a generated molecule
from tdc import Oracle

class TDCDockingEnv(EnvBase): 
    def __init__(self): 
        super().__init__() 
        self.env = "TDC_Docking" # Use the score function provided in TDC. 
        self.oracle = Oracle(name="3pbl_docking") 
       
    def env_forward(self, param): 
        self.parameter_space.check(param) 
        # Construct the input to oracle 
        smiles_string = "".join([param[f"{i}"] for i in range(100)]) 
        return smiles_string 
    
    def score(self, value): 
        # Use oracle to calculate real-valued score. 
        oracle_score = self.oracle(value) 
        _score = np.sum(oracle_score) 
        return _score 
    
    @property 
    def parameter_space(self): 
        # Construct parameter space 
        if not hasattr(self, "_parameter_space"): 
            for i in range(100): 
                # SMILES is used to denote chemical molecular.
                spaces[f"{i}"] =CategorySpace(categories=SMILES_SYMBOLS) 
            self._parameter_space = CombineSpace(space_dict=spaces) 
        return self._parameter_space 
    
    @property 
    def get_input_dim(self): 
        return 100
    
    @property
    def get_output_dim(self):
        return 3
```
## Cite

@inproceedings{IDToolkit,
  author       = {Jia-Qi Yang and
                  Yucheng Xu and
		  Jia-Lei Shen and
                  Kebin Fan and
                  De-Chuan Zhan and
		  Yang Yang},
  title        = {IDToolkit: A Toolkit for Benchmarking and Developing Inverse Design Algorithms in Nanophotonics},
  booktitle    = {{KDD} '23: The 29th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  year         = {2023},
}
