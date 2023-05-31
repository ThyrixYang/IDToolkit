import os
import argparse
from inverse_design_benchmark.envs.multi_layer_env import MultiLayerEnv
from inverse_design_benchmark.envs.tpv_env import TPVEnv
from inverse_design_benchmark.envs.debug_env import DebugEnv, DebugNumEnv
from inverse_design_benchmark.envs.color_filter_env import ColorFilterEnv
import pandas as pd


def gen_multi_layer_data():
    env = MultiLayerEnv()
    params, values = env.generate_dataset(num_data=10)
    params_df = pd.DataFrame(params)
    values_df = pd.DataFrame(values)
    params_df.to_csv("./test_multi_layer_params.csv")
    values_df.to_csv("./test_multi_layer_values.csv")


def gen_tpv_data():
    env = TPVEnv()
    while True:
        env.generate_dataset(num_data=100, save_dir="./tmp/tpv_data")


def gen_debug_data():
    env = DebugEnv()
    env.generate_dataset(num_data=10, save_dir="./tmp/debug_data")

def main(args):
    if args.env == "tpv":
        env = TPVEnv()
    elif args.env == "debug":
        env = DebugEnv()
    elif args.env == "multi_layer":
        env = MultiLayerEnv()
    elif args.env == "debug_num":
        env = DebugNumEnv()
    elif args.env == "color_filter":
        env = ColorFilterEnv()
    else:
        raise ValueError(f"env {args.env} not exists")
    save_dir = os.path.join(args.save_dir, args.env)
    print("Save dir: ", save_dir)
    while True:
        env.generate_dataset(num_data=100,
                             save_dir=save_dir,
                             num_process=args.num_cpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--num_data", type=int, default=100)
    parser.add_argument("--num_cpu", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default="./generated_data")
    args = parser.parse_args()
    main(args)
