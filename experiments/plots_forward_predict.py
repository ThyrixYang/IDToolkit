import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
import yaml

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5})

multi_layer_paths = [
    ("experimental_log/forward_pred/multi_layer/sklearn/linear_regression_/pred_num_10", "linear regression"),
    ("experimental_log/forward_pred/multi_layer/sklearn/decision_tree_/pred_num_10", "decision tree"),
    ("experimental_log/forward_pred/multi_layer/sklearn/xgboost_/pred_num_10", "xgboost"),
    ("experimental_log/forward_pred/multi_layer/neural_network/nn_default+mlp_/pred_num_10", "mlp"),
    ("experimental_log/forward_pred/multi_layer/neural_network/nn_default+cnn_/pred_num_10", "cnn"),
]

tpv_paths = [
    ("experimental_log/forward_pred/tpv/sklearn/linear_regression_/pred_num_10", "linear regression"),
    ("experimental_log/forward_pred/tpv/sklearn/decision_tree_/pred_num_10", "decision tree"),
    ("experimental_log/forward_pred/tpv/sklearn/xgboost_/pred_num_10", "xgboost"),
    ("experimental_log/forward_pred/tpv/neural_network/nn_default+mlp_/pred_num_10", "mlp"),
    ("experimental_log/forward_pred/tpv/neural_network/nn_default+cnn_/pred_num_10", "cnn"),
]

color_filter_paths = [
    ("experimental_log/forward_pred/color_filter/sklearn/linear_regression_/pred_num_10", "linear regression"),
    ("experimental_log/forward_pred/color_filter/sklearn/decision_tree_/pred_num_10", "decision tree"),
    ("experimental_log/forward_pred/color_filter/sklearn/xgboost_/pred_num_10", "xgboost"),
    ("experimental_log/forward_pred/color_filter/neural_network/nn_default+mlp_/pred_num_10", "mlp"),
    ("experimental_log/forward_pred/color_filter/neural_network/nn_default+cnn_/pred_num_10", "cnn"),
]

def load_log_with_different_seeds(log_path):
    train_num_list = os.listdir(log_path)
    train_nums = sorted([int(t.split("_")[-1]) for t in train_num_list])
    all_scores = []
    all_train_nums = []
    for train_num in train_nums:
        train_num_path = os.path.join(log_path, f"train_num_{train_num}")
        seeds = sorted(os.listdir(train_num_path))
        assert len(seeds) > 0
        for seed in seeds:
            file_folder = os.path.join(train_num_path, seed)
            results_path = os.path.join(file_folder, "metrics.yaml")
            with open(results_path, "rb") as f:
                results = yaml.safe_load(f)
            score = results["mse"]
            all_train_nums.append(train_num)
            all_scores.append(score)
    return all_train_nums, all_scores


def plot_inverse_design_methods_with_train_num(log_paths, env):
    for log_path, name in log_paths:
        train_nums, scores = load_log_with_different_seeds(log_path)
        data = pd.DataFrame({"train_nums": train_nums, "scores": scores})
        ax = sns.lineplot(data=data, x="train_nums", y="scores",
                        #   legend="brief", label=name
                          )
        ax.set(yscale='log')
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        # ax.figure.set_size_inches(8.5, 5)
        # ax.set_title(env)
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"experiments/plots/forward_pred_{env}_no_legend.jpg")
        plt.savefig(f"experiments/plots/forward_pred_{env}_no_legend.svg")
        
if __name__ == "__main__":
    plot_inverse_design_methods_with_train_num(tpv_paths, "tpv")
    # plot_inverse_design_methods_with_train_num(multi_layer_paths, "multi_layer")
    # plot_inverse_design_methods_with_train_num(color_filter_paths, "color_filter")