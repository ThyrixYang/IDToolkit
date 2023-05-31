import numpy as np

class Algorithm:
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
    def score_fn(self, param):
        value = self.env.forward(param)
        score = self.env.score(value)
        return {"score": score,
                "value": value,
                "param": param}
        
    def tag_dataset_parameters(self, dataset_parameters):
        outputs = []
        for i, d in enumerate(dataset_parameters):
            d["_train_parameter_index"] = i
            outputs.append(d)
        return outputs
    
    def preprocess(self, num_pred, dataset_parameters):
        if dataset_parameters is not None:
            num_samples = len(dataset_parameters) + num_pred
        else:
            num_samples = num_pred
        return num_samples
    
    def fit_and_search(self,
                       num_pred,
                       dataset_parameters,
                       seed=0):
        raise NotImplementedError()
    
def parse_ray_tune_results(results, num_samples):
    scores = []
    values = []
    params = []
    for r in results:
        scores.append(r.metrics["score"])
        values.append(r.metrics["value"])
        params.append(r.metrics["param"])
    train_params, train_values, train_scores = params[:-num_samples], values[:-num_samples], scores[:-num_samples]
    pred_params, pred_values, pred_scores = params[-num_samples:], values[-num_samples:], scores[-num_samples:]
    res = {
        "pred_scores": pred_scores,
        "pred_values": pred_values,
        "pred_params": pred_params,
        "metrics": {
            "pred_score_max": float(np.max(pred_scores)),
            "pred_score_mean": float(np.mean(pred_scores)),
            "pred_score_std": float(np.std(pred_scores)),
            "pred_num": len(pred_scores),
            "all_score_max": float(np.max(scores)),
            "all_score_mean": float(np.mean(scores)),
            "all_score_std": float(np.std(scores)),
            "all_num": len(scores)
        }
    }
    if len(train_params) > 0:
        res.update({
            "train_scores": train_scores,
            "train_values": train_values,
            "train_params": train_params,
        })
        res["metrics"].update({
            "train_score_max": float(np.max(train_scores)),
            "train_score_mean": float(np.mean(train_scores)),
            "train_score_std": float(np.std(train_scores)),
            "train_num": len(train_scores)
        })
    return res