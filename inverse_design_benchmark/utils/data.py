import os
import multiprocessing
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def load_data(data_input):
    idx, param_fn, dataset_path = data_input
    suffix = param_fn.split("_")[-1]
    values_file_name = f"values_{suffix}"
    params_df = pd.read_csv(Path(dataset_path) / param_fn)
    values_df = pd.read_csv(Path(dataset_path) / values_file_name)
    return (idx, params_df, values_df)
    

def load_dataset(dataset_path):
    file_names = os.listdir(dataset_path)
    params_file_names = sorted([fn for fn in file_names if "params" in fn])
    print("loading data")
    data_inputs = [(i, pfn, dataset_path) for i, pfn in enumerate(params_file_names)]
    # ordered_inputs = [(p, o) for p, o in zip(params, range(len(params)))]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap_unordered(load_data, data_inputs), 
            total=len(params_file_names)))
    results = sorted(results, key=lambda x: x[0])
    params_df = [r[1] for r in results]
    values_df = [r[2] for r in results]
    # values = [x[0] for x in results]
    
    # for pfn in tqdm(params_file_names):
    #     suffix = pfn.split("_")[-1]
    #     values_file_name = f"values_{suffix}"
    #     params_df.append(pd.read_csv(Path(dataset_path) / pfn))
    #     values_df.append(pd.read_csv(Path(dataset_path) / values_file_name))
    params_df = pd.concat(params_df)
    values_df = pd.concat(values_df)
    params_df.drop(params_df.columns[0], axis=1, inplace=True)
    values_df.drop(values_df.columns[0], axis=1, inplace=True)
    params = params_df.to_dict("records")
    
    values = list(values_df.to_numpy())
    return params, values

if __name__ == "__main__":
    load_dataset("/home/yangjq/Working/inverse_design_benchmark/datasets/debug")