import ray
from ray import tune
from ray.tune.search.nevergrad import NevergradSearch
import nevergrad as ng
from ray.air import RunConfig

from .opt_base import Algorithm, parse_ray_tune_results


class OnePlusOneAlgorithm(Algorithm):

    def __init__(self,
                 env,
                 config):
        super().__init__(env=env, config=config)


    def fit_and_search(self,
                       num_pred=1,
                       dataset_parameters=None,
                       seed=0):
        
        if dataset_parameters is not None:
            num_samples = len(dataset_parameters) + num_pred
        else:
            num_samples = num_pred

        self.alg = NevergradSearch(
            points_to_evaluate=dataset_parameters,
            optimizer=ng.optimizers.OnePlusOne,
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
