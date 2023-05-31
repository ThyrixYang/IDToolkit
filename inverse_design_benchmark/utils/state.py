import os

import yaml
import joblib

def save_sklearn_checkpoint(path, alg_name, seed, model):
    dir_name = f'{path}/{alg_name}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = f'{dir_name}/seed_{seed}.ckpt'
    joblib.dump(model, file_name)

def save_std_and_mean(std, mean, path):
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = f'{path}/std_mean.yaml'
    if not os.path.exists(file_name):
        mean_and_std = {
            "std": std.numpy().tolist(),
            "mean": mean.numpy().tolist()
        }
        with open(file_name, "w") as f:
            yaml.dump(mean_and_std, f, default_flow_style=False)
        print(f"Mean and std saved at {file_name}")
    else:
        print(f"Mean and std file exists at {file_name}")