import seaborn as sns
import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5})

color_palette = list(sns.color_palette("deep")) + list(sns.color_palette("husl", 11))
id_methods = ["bayesopt", "hyperopt", "random search", "zoopt", "oneplusone", 
              "cgan", "cvae", "tandem", "gd", "reverse_model", "train_best"]

id_color_map = {name: c for name, c in zip(id_methods, color_palette)}

debug_id_with_pred_num_paths = [
    ("experimental_log/real_target/debug_num/bayesopt/_/pred_num_100/train_num_0", "bayesopt"),
    ("experimental_log/real_target/debug_num/hyperopt/_/pred_num_100/train_num_0", "hyperopt"),
    ("experimental_log/real_target/debug_num/random_search/_/pred_num_100/train_num_0", "random search"),
    ("experimental_log/real_target/debug_num/zoopt/zoopt_/pred_num_100/train_num_0", "zoopt"),
]

multi_layer_id_with_pred_num_paths = [
    ("experimental_log/real_target/multi_layer/bayesopt/bayesopt_/pred_num_1000/train_num_0", "bayesopt"),
    ("experimental_log/real_target/multi_layer/hyperopt/hyperopt_/pred_num_1000/train_num_0", "hyperopt"),
    ("experimental_log/real_target/multi_layer/random_search/random_search_/pred_num_1000/train_num_0", "random search"),
    ("experimental_log/real_target/multi_layer/zoopt/zoopt_/pred_num_1000/train_num_0", "zoopt"),
    ("experimental_log/real_target/multi_layer/oneplusone/oneplusone_/pred_num_1000/train_num_0", "oneplusone"),
]

tpv_id_0_train_with_pred_num_paths = [
    ("experimental_log/real_target/tpv/bayesopt/bayesopt_/pred_num_200/train_num_0", "bayesopt"),
    ("experimental_log/real_target/tpv/hyperopt/hyperopt_/pred_num_200/train_num_0", "hyperopt"),
    ("experimental_log/real_target/tpv/random_search/random_search_/pred_num_200/train_num_0", "random search"),
    ("experimental_log/real_target/tpv/zoopt/zoopt_/pred_num_200/train_num_0", "zoopt"),
    ("experimental_log/real_target/tpv/oneplusone/oneplusone_/pred_num_200/train_num_0", "oneplusone"),
]

tpv_id_all_train_with_pred_num_paths = [
    ("experimental_log/real_target/tpv/bayesopt/bayesopt_/pred_num_200/train_num_100", "bayesopt"),
    ("experimental_log/real_target/tpv/hyperopt/hyperopt_/pred_num_200/train_num_100", "hyperopt"),
    ("experimental_log/real_target/tpv/random_search/random_search_/pred_num_200/train_num_100", "random search"),
    # ("experimental_log/real_target/tpv/oneplusone/oneplusone_/pred_num_200/train_num_100", "oneplusone"),
    
    ("experimental_log/real_target/tpv/cgan/cgan_/pred_num_200/train_num_1000000", "cgan"),
    ("experimental_log/real_target/tpv/cvae/cvae_/pred_num_200/train_num_1000000", "cvae"),
    ("experimental_log/real_target/tpv/gd/gd_/pred_num_200/train_num_1000000", "gd"),
    ("experimental_log/real_target/tpv/reverse_model/reverse_model_/pred_num_200/train_num_1000000", "reverse_model"),
    ("experimental_log/real_target/tpv/tandem/tandem_/pred_num_200/train_num_1000000", "tandem"),
]

tpv_iid_eval_0_train_with_pred_num_paths = [
    ("experimental_log/iid_target/tpv/bayesopt/bayesopt_/pred_num_200/train_num_0", "bayesopt"),
    ("experimental_log/iid_target/tpv/hyperopt/hyperopt_/pred_num_200/train_num_0", "hyperopt"),
    ("experimental_log/iid_target/tpv/random_search/random_search_/pred_num_200/train_num_0", "random search"),
    ("experimental_log/iid_target/tpv/oneplusone/oneplusone_/pred_num_200/train_num_0", "oneplusone"),
    ("experimental_log/iid_target/tpv/zoopt/zoopt_/pred_num_200/train_num_0", "zoopt"),
]

tpv_iid_eval_all_train_with_pred_num_paths = [
    ("experimental_log/iid_target/tpv/bayesopt/bayesopt_/pred_num_200/train_num_100", "bayesopt"),
    ("experimental_log/iid_target/tpv/hyperopt/hyperopt_/pred_num_200/train_num_100", "hyperopt"),
    ("experimental_log/iid_target/tpv/random_search/random_search_/pred_num_200/train_num_100", "random search"),
    # ("experimental_log/iid_target/tpv/oneplusone/oneplusone_/pred_num_200/train_num_100", "oneplusone"),
    # ("experimental_log/real_target/tpv/zoopt/zoopt_/pred_num_200/train_num_100", "zoopt"),
    ("experimental_log/iid_target/tpv/cgan/cgan_/pred_num_200/train_num_1000000", "cgan"),
    ("experimental_log/iid_target/tpv/cvae/cvae_/pred_num_200/train_num_1000000", "cvae"),
    ("experimental_log/iid_target/tpv/gd/gd_/pred_num_200/train_num_1000000", "gd"),
    ("experimental_log/iid_target/tpv/reverse_model/reverse_model_/pred_num_200/train_num_1000000", "reverse_model"),
    ("experimental_log/iid_target/tpv/tandem/tandem_/pred_num_200/train_num_1000000", "tandem"),
]

multi_layer_id_0_train_with_pred_num_paths = [
    ("experimental_log/real_target/multi_layer/bayesopt/bayesopt_/pred_num_1000/train_num_0", "bayesopt"),
    ("experimental_log/real_target/multi_layer/hyperopt/hyperopt_/pred_num_1000/train_num_0", "hyperopt"),
    ("experimental_log/real_target/multi_layer/random_search/random_search_/pred_num_1000/train_num_0", "random search"),
    ("experimental_log/real_target/multi_layer/oneplusone/oneplusone_/pred_num_1000/train_num_0", "oneplusone"),
    ("experimental_log/real_target/multi_layer/zoopt/zoopt_/pred_num_1000/train_num_0", "zoopt"),
]

multi_layer_id_all_train_with_pred_num_paths = [
    ("experimental_log/real_target/multi_layer/bayesopt/bayesopt_/pred_num_1000/train_num_100", "bayesopt"),
    ("experimental_log/real_target/multi_layer/hyperopt/hyperopt_/pred_num_1000/train_num_100", "hyperopt"),
    ("experimental_log/real_target/multi_layer/random_search/random_search_/pred_num_1000/train_num_100", "random search"),
    # ("experimental_log/real_target/multi_layer/oneplusone/oneplusone_/pred_num_1000/train_num_100", "oneplusone"),
    # ("experimental_log/real_target/multi_layer/zoopt/zoopt_/pred_num_1000/train_num_0", "zoopt"),
]

multi_layer_iid_eval_all_train_with_pred_num_paths = [
    ("experimental_log/iid_target/multi_layer/bayesopt/bayesopt_/pred_num_1000/train_num_100", "bayesopt"),
    ("experimental_log/iid_target/multi_layer/hyperopt/hyperopt_/pred_num_1000/train_num_100", "hyperopt"),
    ("experimental_log/iid_target/multi_layer/random_search/random_search_/pred_num_1000/train_num_100", "random search"),
    # ("experimental_log/iid_target/multi_layer/oneplusone/oneplusone_/pred_num_1000/train_num_100", "oneplusone"),
    # ("experimental_log/real_target/multi_layer/zoopt/zoopt_/pred_num_1000/train_num_0", "zoopt"),
]

multi_layer_iid_eval_0_train_with_pred_num_paths = [
    ("experimental_log/iid_target/multi_layer/bayesopt/bayesopt_/pred_num_1000/train_num_0", "bayesopt"),
    ("experimental_log/iid_target/multi_layer/hyperopt/hyperopt_/pred_num_1000/train_num_0", "hyperopt"),
    ("experimental_log/iid_target/multi_layer/random_search/random_search_/pred_num_1000/train_num_0", "random search"),
    ("experimental_log/iid_target/multi_layer/oneplusone/oneplusone_/pred_num_1000/train_num_0", "oneplusone"),
    ("experimental_log/iid_target/multi_layer/zoopt/zoopt_/pred_num_1000/train_num_0", "zoopt"),
]

color_filter_0_train_with_pred_num_paths = [
    ("experimental_log/real_target/color_filter/bayesopt/bayesopt_/pred_num_200/train_num_0", "bayesopt"),
    ("experimental_log/real_target/color_filter/hyperopt/hyperopt_/pred_num_200/train_num_0", "hyperopt"),
    ("experimental_log/real_target/color_filter/random_search/random_search_/pred_num_200/train_num_0", "random search"),
    ("experimental_log/real_target/color_filter/oneplusone/oneplusone_/pred_num_200/train_num_0", "oneplusone"),
    ("experimental_log/real_target/color_filter/zoopt/zoopt_/pred_num_200/train_num_0", "zoopt"),
]

color_filter_all_train_with_pred_num_paths = [
    ("experimental_log/real_target/color_filter/bayesopt/bayesopt_/pred_num_200/train_num_100", "bayesopt"),
    ("experimental_log/real_target/color_filter/hyperopt/hyperopt_/pred_num_200/train_num_100", "hyperopt"),
    ("experimental_log/real_target/color_filter/random_search/random_search_/pred_num_200/train_num_100", "random search"),
    
    ("experimental_log/real_target/color_filter/cgan/cgan_/pred_num_200/train_num_1000000", "cgan"),
    ("experimental_log/real_target/color_filter/cvae/cvae_/pred_num_200/train_num_1000000", "cvae"),
    ("experimental_log/real_target/color_filter/gd/gd_/pred_num_200/train_num_1000000", "gd"),
    ("experimental_log/real_target/color_filter/reverse_model/reverse_model_/pred_num_200/train_num_1000000", "reverse_model"),
    ("experimental_log/real_target/color_filter/tandem/tandem_/pred_num_200/train_num_1000000", "tandem"),
]

color_filter_iid_0_train_with_pred_num_paths = [
    ("experimental_log/iid_target/color_filter/bayesopt/bayesopt_/pred_num_200/train_num_0", "bayesopt"),
    ("experimental_log/iid_target/color_filter/hyperopt/hyperopt_/pred_num_200/train_num_0", "hyperopt"),
    ("experimental_log/iid_target/color_filter/random_search/random_search_/pred_num_200/train_num_0", "random search"),
    ("experimental_log/iid_target/color_filter/oneplusone/oneplusone_/pred_num_200/train_num_0", "oneplusone"),
    ("experimental_log/iid_target/color_filter/zoopt/zoopt_/pred_num_200/train_num_0", "zoopt"),
]

color_filter_iid_all_train_with_pred_num_paths = [
    ("experimental_log/iid_target/color_filter/bayesopt/bayesopt_/pred_num_200/train_num_100", "bayesopt"),
    ("experimental_log/iid_target/color_filter/hyperopt/hyperopt_/pred_num_200/train_num_100", "hyperopt"),
    ("experimental_log/iid_target/color_filter/random_search/random_search_/pred_num_200/train_num_100", "random search"),
    
    ("experimental_log/iid_target/color_filter/cgan/cgan_/pred_num_200/train_num_1000000", "cgan"),
    ("experimental_log/iid_target/color_filter/cvae/cvae_/pred_num_200/train_num_1000000", "cvae"),
    ("experimental_log/iid_target/color_filter/gd/gd_/pred_num_200/train_num_1000000", "gd"),
    ("experimental_log/iid_target/color_filter/reverse_model/reverse_model_/pred_num_200/train_num_1000000", "reverse_model"),
    ("experimental_log/iid_target/color_filter/tandem/tandem_/pred_num_200/train_num_1000000", "tandem"),
]


def load_log_with_different_seeds(log_path):
    seeds = sorted(os.listdir(log_path))
    
    pred_num = int(log_path[log_path.find("pred_num")+9:].split("/")[0])
    method = log_path.split("/")[3]
    
    all_scores = []
    all_trial_ids = []
    min_length = 10000000000
    assert len(seeds) > 0
    for seed in seeds:
        file_folder = os.path.join(log_path, seed)
        results_path = os.path.join(file_folder, "results_final.pkl")
        if not os.path.isfile(results_path):
            continue
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        scores = np.array(results["pred_scores"])
        if scores.shape[0] == 1:
            scores = np.repeat(scores, pred_num)
        for i in range(len(scores) - 1):
            scores[i + 1] = max(scores[i + 1], scores[i])
        scores = -scores
        trial_id = np.array(list(range(len(scores))))
        min_length = min(min_length, len(scores))
        all_scores.append(scores)
        all_trial_ids.append(trial_id)
    for i in range(len(all_scores)):
        all_scores[i] = all_scores[i][:min_length]
        all_trial_ids[i] = all_trial_ids[i][:min_length]
    all_trial_ids = np.concatenate(all_trial_ids)
    all_scores = np.concatenate(all_scores)
    return all_trial_ids, all_scores

def load_train_score(log_path):
    seeds = sorted(os.listdir(log_path))
    ids = []
    scores = []
    for seed in seeds:
        file_folder = os.path.join(log_path, seed)
        results_path = os.path.join(file_folder, "metrics.yaml")
        with open(results_path, "r") as f:
            results = yaml.safe_load(f)
        pred_num = results["pred_num"]
        train_score_max = results["train_score_max"]
        ids.append(np.arange(pred_num))
        scores.append(-train_score_max * np.ones(pred_num))
    ids = np.concatenate(ids)
    scores = np.concatenate(scores)
    # print(scores)
    # exit()
    # print(ids.shape)
    # print(scores.shape)
    # exit()
    return scores, ids

def plot_inverse_design_methods_with_pred_num(
        log_paths,
        file_name):
    
    sample_path = log_paths[-1][0]
    train_num = int(sample_path[sample_path.find("train_num")+10:])
    
    for log_path, name in log_paths:
            
        trial_ids, scores = load_log_with_different_seeds(log_path)
        data = pd.DataFrame({"trials": trial_ids, "scores": scores})
        ax = sns.lineplot(data=data, x="trials", y="scores",
                        #   legend="brief", label=name, 
                          color=id_color_map[name])
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        if train_num > 0:
            train_score, train_id = load_train_score(log_path)
            train_num = 0
            data = pd.DataFrame({"trials": train_id, "scores": train_score})
            ax = sns.lineplot(data=data, x="trials", y="scores",
                            # legend="brief", label="train_best", 
                            color=id_color_map["train_best"],
                            linestyle='--')
            ax.set(xlabel=None)
            ax.set(ylabel=None)
    
    ax.set(yscale='log')
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig("experiments/plots/{}_no_legend.jpg".format(file_name))
    plt.savefig("experiments/plots/{}_no_legend.pdf".format(file_name))
    plt.savefig("experiments/plots/{}_no_legend.svg".format(file_name))
    plt.clf()
    plt.cla()
    # x = np.array(list(range(10)))
    # x = np.concatenate([x, x, x], axis=0)
    # y = np.random.randn(30)
    # data = pd.DataFrame({"x": x, "y": y})
    # sns.lineplot(data=data, x="x", y="y")
    # plt.tight_layout()
    # plt.savefig("experiments/plots/test.jpg")


def test():
    sns.set_theme(style="darkgrid")

    # Load an example dataset with long-form data
    # fmri = sns.load_dataset("fmri")
    fmri = pd.read_csv("experiments/tmp/seaborn-data-master/fmri.csv")

    # Plot the responses for different events and regions
    sns.lineplot(x="timepoint", y="signal",
                 hue="region", style="event",
                 data=fmri)
    plt.savefig("experiments/plots/test.jpg")


if __name__ == "__main__":
    # test()
    # plot_inverse_design_methods_with_pred_num(0, 0)
    # plot_inverse_design_methods_with_pred_num(log_paths=id_with_pred_num_paths)
    # plot_inverse_design_methods_with_pred_num(log_paths=multi_layer_id_with_pred_num_paths)
    # plot_inverse_design_methods_with_pred_num(
    #     log_paths=tpv_id_0_train_with_pred_num_paths,
    #     file_name="tpv_0_train")
    # plot_inverse_design_methods_with_pred_num(
    #     log_paths=tpv_id_all_train_with_pred_num_paths,
    #     file_name="tpv_all_train")
    # plot_inverse_design_methods_with_pred_num(
    #     log_paths=tpv_iid_eval_0_train_with_pred_num_paths,
    #     file_name="tpv_iid_0_train")
    # plot_inverse_design_methods_with_pred_num(
    #     log_paths=tpv_iid_eval_all_train_with_pred_num_paths,
    #     file_name="tpv_iid_all_train")
    # plot_inverse_design_methods_with_pred_num(
    #     log_paths=multi_layer_id_0_train_with_pred_num_paths,
    #     file_name="multi_layer_0_train")
    # plot_inverse_design_methods_with_pred_num(
    #     log_paths=multi_layer_id_all_train_with_pred_num_paths,
    #     file_name="multi_layer_all_train")
    # plot_inverse_design_methods_with_pred_num(
    #     log_paths=multi_layer_iid_eval_all_train_with_pred_num_paths,
    #     file_name="multi_layer_iid_all_train")
    # plot_inverse_design_methods_with_pred_num(
    #     log_paths=multi_layer_iid_eval_0_train_with_pred_num_paths,
    #     file_name="multi_layer_iid_0_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=tpv_id_0_train_with_pred_num_paths,
        file_name="tpv_0_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=tpv_id_all_train_with_pred_num_paths,
        file_name="tpv_all_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=tpv_iid_eval_0_train_with_pred_num_paths,
        file_name="tpv_iid_0_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=tpv_iid_eval_all_train_with_pred_num_paths,
        file_name="tpv_iid_all_train")
    
    plot_inverse_design_methods_with_pred_num(
        log_paths=multi_layer_id_0_train_with_pred_num_paths,
        file_name="multi_layer_0_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=multi_layer_id_all_train_with_pred_num_paths,
        file_name="multi_layer_all_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=multi_layer_iid_eval_all_train_with_pred_num_paths,
        file_name="multi_layer_iid_all_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=multi_layer_iid_eval_0_train_with_pred_num_paths,
        file_name="multi_layer_iid_0_train")
    
    plot_inverse_design_methods_with_pred_num(
        log_paths=color_filter_0_train_with_pred_num_paths,
        file_name="color_filter_real_target_0_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=color_filter_all_train_with_pred_num_paths,
        file_name="color_filter_real_target_all_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=color_filter_iid_all_train_with_pred_num_paths,
        file_name="color_filter_iid_target_all_train")
    plot_inverse_design_methods_with_pred_num(
        log_paths=color_filter_iid_0_train_with_pred_num_paths,
        file_name="color_filter_iid_target_0_train")
