import warnings

from ray import tune
from ray.tune.search.zoopt import ZOOptSearch
from ray.air import RunConfig

from .opt_base import Algorithm, parse_ray_tune_results


class ZOOptAlgorithm(Algorithm):

    def __init__(self,
                 env,
                 config):
        super().__init__(env=env, config=config)

    def fit_and_search(self,
                       num_pred=1,
                       dataset_parameters=None,
                       seed=0):
        num_samples = self.preprocess(num_pred, dataset_parameters)
        if dataset_parameters is not None and len(dataset_parameters) > 0:
            raise ValueError("Dataset is not well supported by zoopt")
            num_samples = len(dataset_parameters) + num_pred
            # dataset_parameters = None
            num_samples = num_pred
        else:
            num_samples = num_pred

        zoopt_search_config = {
            "parallel_num": self.config.parallel_num,  # how many workers to parallel
            # "parallel_num": 16,  # how many workers to parallel
        }
        dim_dict = self.env.parameter_space.to_zooopt_space()

        self.alg = ZOOptSearch(
            algo="Asracos",  # only support Asracos currently
            # must match `num_samples` in `tune.TuneConfig()`.
            budget=num_samples,
            dim_dict=dim_dict,
            metric="score",
            mode="max",
            points_to_evaluate=dataset_parameters,
            **zoopt_search_config
        )

        tuner = tune.Tuner(
            self.score_fn,
            tune_config=tune.TuneConfig(
                search_alg=self.alg,
                metric="score",
                num_samples=num_samples,
                mode="max",
            ),
            run_config=RunConfig(verbose=1),
        )
        ray_results = tuner.fit()
        results = parse_ray_tune_results(ray_results, num_samples=num_pred)
        return results
