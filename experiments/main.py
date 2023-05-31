import argparse
import time
import numpy as np
from rich import print

from inverse_design_benchmark.utils import load_dataset

from utils import get_alg, get_env, save_experiment_results
from utils import set_seed, check_results_exists, evaluate_predicted_parameters, evaluate_intermediate_for_substitute_models
from utils import evaluate_predicted_values, update_config
from config_tool import load_config


def main(args):
    alg_fn = get_alg(args.method)
    env_fn = get_env(args.env)
    env = env_fn(args.seed, args.save_substitute_model, args.substitute_model, args.ensemble)
    method_config = load_config(args.method_config)
    method_config = update_config(args.alg_args, method_config)
    print("================ method config ===============")
    print(f"{method_config}")
    print("=============================================")
    
    # the final result for the task already exists!

    if check_results_exists(args, tag="final"):
        print("Final results exist, exit")
        return

    alg = alg_fn(env, method_config)

    data_path = args.dataset_path
    if data_path is not None:
        data_params, data_values = load_dataset(data_path)
        test_data_num = int(args.test_ratio * len(data_params))
        test_params, test_values = data_params[:
                                            test_data_num], data_values[:test_data_num]

    if args.eval_method == "forward_pred":
        train_params, train_values \
            = data_params[test_data_num:][:args.train_num], data_values[test_data_num:][:args.train_num]
        print(f"Train data len [{len(train_params)}]")
        alg.fit(train_params, train_values)
        pred_values = alg.predict(test_params)
        test_values = np.stack(test_values)
        results = evaluate_predicted_values(pred_values, test_values)
        save_experiment_results(args, method_config, results, tag="final")
    elif args.eval_method == "real_target" or args.eval_method == "iid_target":
        # iid targets are generated randomly from value space
        if args.eval_method == "iid_target":
            print(f"Modify the target to a IID data {args.seed}")
            env.target = test_values[args.seed]
        
        # real targets are set by experts with domain knowledge

        if args.eval_method == "real_target":
            if args.env == "color_filter":
                print(f"Using the {args.seed}th target")
                env.target = env.targets[args.seed]
        
        # these are iterative optimizers
        if args.method in ["random_search", "hyperopt", "bayesopt", "zoopt", "oneplusone"]:
            _train_params, _train_values \
                = data_params[test_data_num:], data_values[test_data_num:]
            _train_scores = [-env.score(v) for v in _train_values]
            train_index = np.argsort(_train_scores)[:args.train_num]
            train_params = [_train_params[i] for i in train_index]
            train_values = [_train_values[i] for i in train_index]
            print(f"Train data len [{len(train_params)}]")
            env.init_dataset(train_params, train_values)

            # train data is non-essential for some iterative optimizers
            # see what is returned from
            #     inverse_design_benchmark/algorithms/opt_base.py: parse_ray_tune_results()
            results = alg.fit_and_search(
                num_pred=args.pred_num,
                dataset_parameters=train_params,
                seed=args.seed)
            
            # with substitute models, iterative optimizers implemented by ray tune does not produce precise values/scores
            if args.substitute_model:
                save_experiment_results(args, method_config, results, tag="intermediate")
                # evaluate the intermediate results from last step.
                #     it's quite slow, so a config is required to do it immediately
                if args.evaluate_after_search:
                    real_results = evaluate_intermediate_for_substitute_models(args, env)
                    save_experiment_results(args, method_config, real_results, tag='final')
            else:
                # it's far more slow, and sadly nothing helps
                save_experiment_results(args, method_config, results, tag="final")

        # these are deep inverse design methods
        elif args.method in ["cvae", "cgan", "gd", "tandem", "reverse_model"]:
            train_params, train_values \
                = data_params[test_data_num:][:args.train_num], data_values[test_data_num:][:args.train_num]
            print(f"Train data len [{len(train_params)}]")
            if check_results_exists(args, tag="pred_params"):
                # print("only prediction")
                # return
                print("start forward simulation")
                results = evaluate_predicted_parameters(args=args, env=env)
                save_experiment_results(args, None, results, tag="final")
                return
            alg.fit(train_params, train_values)
            pred_params = alg.search(num_samples=args.pred_num)
            results = {"pred_params": pred_params,
                       "train_params": train_params,
                       "train_values": train_values}
            # deep inverse design methods directly produce solutions (pred_params) without the usage of simulator,
            #   so pred_params must be evaluated to give metrics
            save_experiment_results(args, method_config, results, tag="pred_params")
            results = evaluate_predicted_parameters(args=args, env=env)
            save_experiment_results(args, None, results, tag="final")
        else:
            raise ValueError()
    else:
        raise ValueError(f"args.eval_method {args.eval_method} not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # the config file of the specified algorithm
    #    could be <full-path-to-config>+<other-config-in-same-dir>
    #    example: --method_config "experiments/configs/nn_default+cnn"
    parser.add_argument("--method_config", type=str, default="")

    # the specified environment 
    parser.add_argument("--env", type=str,
                        choices=["debug",
                                 "tpv",
                                 "multi_layer",
                                 "debug_num",
                                 "color_filter"])
    
    # the path where dataset is stored
    #    a dir of many files named after params-<hashkey>.csv or values-<hashkey>.csv
    parser.add_argument("--dataset_path", type=str, default=None)
    
    # which problem to solve
    parser.add_argument("--eval_method", type=str,
                        choices=[
                            "iid_target",
                            "forward_pred",
                            "real_target"
                        ])
    
    # how many solutions for one target value
    parser.add_argument("--pred_num", type=int, default=10)

    # random seeds for all related packages
    #     multiple values are separated by ","
    parser.add_argument("--seeds", type=str, default="0")

    parser.add_argument("--random_target_num", type=int)
    parser.add_argument("--log_path", type=str, default="./experimental_log")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--train_num", type=int, default=100000)
    parser.add_argument("-d", "--delete_exist", action="store_true")
    parser.add_argument("--alg_args", type=str, default="")

    # in forward prediction, it's the predictor used, could be
    #    neural_networks: deep neural networks based on pytorch_lightning interface
    #    sklearn: other algorithms based on sklearn interface
    # in inverse design, it specifies the algorithm
    parser.add_argument("--method", type=str)

    # arguments related to substitute models
    
    # save the checkpoints when training forward prediction models
    #     it could be used as one substistute model for numerical simluator with iterative optimizers such as random-search
    #     resulte are saved at checkpoint/{env_name}/{alg}
    parser.add_argument("--save_substitute_model", default=False, action='store_true')

    # specify the substitute model which is used to accelarate iterative optimizers
    #    only used in inverse design problems
    #    default value is '', which means no substitute model is used, numerical simulator is used
    parser.add_argument("--substitute_model", type=str, default='')

    # when using substitue model, this flag ensemble all models with same name but different seed
    #    otherwise, only one model is used
    parser.add_argument("--ensemble", action='store_true')

    # evalute the intermediate results immdiately after the inverse design params are searched by iterative optimizers with substitute models
    parser.add_argument("--evaluate_after_search", default=False, action="store_true")
    
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        start_time = time.time()
        set_seed(seed)
        args.seed = seed
        print(args)
        main(args)
        end_time = time.time()
        print(f"Run time [{end_time - start_time} seconds]")
