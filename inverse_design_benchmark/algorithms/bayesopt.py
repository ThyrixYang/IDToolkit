import ray
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.air import RunConfig

from .opt_base import Algorithm, parse_ray_tune_results


class BayesOptAlgorithm(Algorithm):

    def __init__(self,
                 env,
                 config):
        super().__init__(env=env, config=config)
        
    def score_fn(self, param):
        decode_param = self.env.parameter_space.uniform_decode(param)
        value = self.env.forward(decode_param)
        score = self.env.score(value)
        return {"score": score,
                "value": value,
                "param": param}

    def fit_and_search(self,
                       num_pred=1,
                       dataset_parameters=None,
                       seed=0):

        if dataset_parameters is not None:
            dataset_parameters = [
                self.env.parameter_space.uniform_encode(p) 
                for p in dataset_parameters]
            num_samples = len(dataset_parameters) + num_pred
        else:
            num_samples = num_pred
            
        self.alg = BayesOptSearch(
            points_to_evaluate=dataset_parameters,
            random_state=seed
        )
        tuner = tune.Tuner(
            self.score_fn,
            tune_config=tune.TuneConfig(
                search_alg=self.alg,
                metric="score",
                num_samples=num_samples,
                mode="max",
            ),
            param_space=self.env.parameter_space.to_ray_space(continuous=True),
            run_config=RunConfig(verbose=1),
        )
        ray_results = tuner.fit()
        results = parse_ray_tune_results(ray_results, num_pred)
        return results
