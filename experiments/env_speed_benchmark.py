import os
import time
import argparse
from inverse_design_benchmark.envs.multi_layer_env import MultiLayerEnv
from inverse_design_benchmark.envs.tpv_env import TPVEnv
from inverse_design_benchmark.envs.color_filter_env import ColorFilterEnv
import pandas as pd
    
def main(args):
    if args.env == "tpv":
        env = TPVEnv()
    elif args.env == "multi_layer":
        env = MultiLayerEnv()
    elif args.env == "color_filter":
        env = ColorFilterEnv()
    else:
        raise ValueError(f"env {args.env} not exists")
    start_time = time.time()
    env.generate_dataset(num_data=100, 
                            save_dir=None,
                            num_process=args.num_process)
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    print(f"Env name {args.env}, num_process {args.num_process}, total time used {end_time - start_time}, avg time {avg_time}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--num_process", type=int)
    args = parser.parse_args()
    # main(args)
    for np in [2, 4, 8, 16]:
        args.num_process = np
        main(args)
