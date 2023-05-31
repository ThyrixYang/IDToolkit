import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import RunConfig

from .opt_base import Algorithm, parse_ray_tune_results


class HyperOptAlgorithm(Algorithm):

    def __init__(self,
                 env,
                 config):
        super().__init__(env=env, config=config)


    def fit_and_search(self,
                       num_pred=1,
                       dataset_parameters=None,
                       seed=0):
        num_samples = self.preprocess(num_pred, dataset_parameters)
        
        self.alg = HyperOptSearch(
            points_to_evaluate=dataset_parameters,
            random_state_seed=seed
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
            run_config=RunConfig(verbose=1),
        )
        ray_results = tuner.fit()
        results = parse_ray_tune_results(ray_results, num_samples=num_pred)
        return results
