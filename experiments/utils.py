import os
import pathlib
import pickle
import yaml
import numpy as np
import torch
import shutil
from rich import print

from inverse_design_benchmark.algorithms import *
from inverse_design_benchmark.envs import DebugEnv, TPVEnv, MultiLayerEnv, DebugNumEnv, ColorFilterEnv

env_mapper = {
    "debug": DebugEnv,
    "tpv": TPVEnv,
    "multi_layer": MultiLayerEnv,
    "debug_num": DebugNumEnv,
    "color_filter": ColorFilterEnv,
}

from config_tool import config_to_dict



def is_remote_run():
    if "IN_RRUN" in os.environ:
        return True
    else:
        return False


def get_alg(alg_name):
    alg_mapper = {
        "hyperopt": HyperOptAlgorithm,
        "bayesopt": BayesOptAlgorithm,
        "random_search": RandomSearchAlgorithm,
        "zoopt": ZOOptAlgorithm,
        "oneplusone": OnePlusOneAlgorithm,
        "sklearn": SklearnPredictor,
        "neural_network": NeuralNetworkPredictor,
        "cvae": NeuralOptAlgorithm,
        "cgan": NeuralOptAlgorithm,
        "gd": NeuralOptAlgorithm,
        "reverse_model": NeuralOptAlgorithm,
        "tandem": NeuralOptAlgorithm
    }
    if alg_name not in alg_mapper:
        raise ValueError(f"Algorithm [{alg_name}] not exists")
    return alg_mapper[alg_name]


def get_env(env_name):
    if env_name not in env_mapper:
        raise ValueError(f"Env [{env_name}] not exists")
    return env_mapper[env_name]


def get_save_path(args, tag, create=False):
    # create log path if not exists
    if args.substitute_model:
        eval_method = f'{args.eval_method}_sub'
        method = f'{args.method}/{args.substitute_model}'
        if args.ensemble:
            method = f'{method}_ensemble'
    else:
        eval_method = args.eval_method
        method = args.method
    log_path = pathlib.Path(args.log_path) / eval_method / args.env \
        / method / "{}_{}".format(args.method_config.split("/")[-1], args.alg_args) \
        / f"pred_num_{args.pred_num}" / f"train_num_{args.train_num}" / str(args.seed)
    if create:
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    results_path = str((log_path / f"results_{tag}.pkl").resolve())
    return log_path, results_path


def check_results_exists(args, tag):
    log_path, results_path = get_save_path(args, tag, create=False)
    if os.path.isdir(log_path):
        if args.delete_exist:
            shutil.rmtree(log_path)
            print(f"Existing results at [{log_path}] deleted")
    return os.path.isfile(results_path)


def save_experiment_results(args, method_config, results, tag):
    log_path, results_path = get_save_path(args, tag, create=True)
    args_dict = vars(args)
    args_path = str((log_path / "args.yaml").resolve())
    method_config_path = str((log_path / "method_config.yaml").resolve())
    metrics_path = str((log_path / "metrics.yaml").resolve())
    with open(args_path, "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    if method_config is not None:
        with open(method_config_path, "w") as f:
            yaml.dump(config_to_dict(method_config),
                      f, default_flow_style=False)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"results saved at {log_path}")
    if tag == "final":
        print("results.metrics ", results["metrics"])
        with open(metrics_path, "w") as f:
            yaml.dump(results["metrics"], f, default_flow_style=False)


def evaluate_intermediate_for_substitute_models(args, env):
    '''
            This function is used to re-evaluate values and scores got by iterative optimizer 
        with substitute models. Only the numerical simulation is the precise measure of the 
        inverse design solution("pred_params" and "train_params"), so the values and scores 
        from last step are not very sound. Re-evaluate them and produce final metrics.
    '''
    _, data_path = get_save_path(args, tag="intermediate", create=True)
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    pred_params = data["pred_params"]
    pred_values = env.batch_forward(pred_params)
    pred_scores = [env.score(v) for v in pred_values]

    results = {
        "pred_scores": pred_scores,
        "pred_values": pred_values,
        "pred_params": pred_params,
        "metrics": {
            "pred_score_max": float(np.max(pred_scores)),
            "pred_score_mean": float(np.mean(pred_scores)),
            "pred_score_std": float(np.std(pred_scores)),
            "pred_num": len(pred_scores),
            "all_num": data["metrics"]["all_num"]
        }
    }

    if "train_params" in data:
        train_params = data["train_params"]
        train_values = env.batch_forward(train_params)
        train_scores = [env.score(v) for v in train_values]
        results.update({
            "train_params": train_params,
            "train_values": train_values,
            "train_scores": train_scores
        })

        all_score = pred_scores + train_scores
        results["metrics"].update({
            "train_score_max": float(np.max(train_scores)),
            "train_score_mean": float(np.mean(train_scores)),
            "train_score_std": float(np.std(train_scores)),
        })
    else:
        all_score = pred_scores
    results["metrics"].update({
        "all_score_max": float(np.max(all_score)),
        "all_score_mean": float(np.mean(all_score)),
        "all_score_std": float(np.std(all_score))
    })
    return results

def evaluate_predicted_parameters(args, env):
    _, params_path = get_save_path(args, tag="pred_params", create=True)
    with open(params_path, "rb") as f:
        pred_data = pickle.load(f)
    pred_params = pred_data["pred_params"]
    train_params = pred_data["train_params"]
    train_values = pred_data["train_values"]
    pred_values = env.batch_forward(pred_params)
    pred_scores = [env.score(v) for v in pred_values]
    train_scores = [env.score(v) for v in train_values]
    results = {
        "pred_scores": pred_scores,
        "pred_values": pred_values,
        "pred_params": pred_params,
        "train_scores": train_scores,
        "train_params": train_params,
        "train_values": train_values,
        "metrics": {
            "train_score_max": float(np.max(train_scores)),
            "train_score_mean": float(np.mean(train_scores)),
            "train_score_std": float(np.std(train_scores)),
            "pred_score_max": float(np.max(pred_scores)),
            "pred_score_mean": float(np.mean(pred_scores)),
            "pred_score_std": float(np.std(pred_scores)),
            "all_score_max": float(np.max(train_scores + pred_scores)),
            "all_score_mean": float(np.mean(train_scores + pred_scores)),
            "all_score_std": float(np.std(train_scores + pred_scores))
        }
    }
    return results


def evaluate_predicted_values(pred_values, test_values):
    mse = np.mean((pred_values - test_values)**2)
    mae = np.mean(np.abs(pred_values - test_values))
    max_error = np.mean(np.max(np.abs(pred_values - test_values), axis=1))
    results = {
        "pred_values": pred_values,
        "test_values": test_values,
        "metrics": {
            "mse": float(mse),
            "mae": float(mae),
            "mean_max_error": float(max_error)
        }
    }
    return results


def update_config(alg_args, config):
    alg_args = alg_args.split()
    assert len(alg_args) % 2 == 0
    updates = {}
    for i in range(len(alg_args) // 2):
        updates[alg_args[i * 2].strip("-")] = alg_args[i * 2 + 1]
    config.update(updates)
    return config


def set_seed(seed):
    print(f"set seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
