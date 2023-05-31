import ray
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.air import RunConfig

from .opt_base import Algorithm, parse_ray_tune_results


class RandomSearchAlgorithm(Algorithm):

    def __init__(self,
                 env,
                 config):
        super().__init__(env=env, config=config)
        
    def score_fn(self, param):
        value = self.env.forward(param)
        score = self.env.score(value)
        return {"score": score,
                "value": value,
                "param": param}

    def fit_and_search(self,
                       num_pred=1,
                       dataset_parameters=None,
                       seed=0):
        num_samples = self.preprocess(num_pred, dataset_parameters)
        self.alg = BasicVariantGenerator(
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
            param_space=self.env.parameter_space.to_ray_space(continuous=False),
            run_config=RunConfig(verbose=1),
        )
        ray_results = tuner.fit()
        results = parse_ray_tune_results(ray_results, num_samples=num_pred)
        return results
