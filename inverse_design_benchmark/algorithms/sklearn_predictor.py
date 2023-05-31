import os
from xgboost import XGBRegressor as XGBR
from xgboost.callback import _Model, TrainingCallback
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from .predictor_base import Predictor
from ..utils.state import save_sklearn_checkpoint

class MyXgboostTopOneCheckpointCallback(TrainingCallback):
    def __init__(self, dest_dir: str, seed: int) -> None:
        super().__init__()
        self.dest_dir = dest_dir
        self.seed = seed
        self.best_model = None
        self.best_score = None

    def after_iteration(self, model, epoch, evals_log) -> bool:
        if self.best_model is None:
            assert self.best_score is None
            self.best_model = model
            self.best_score = evals_log['validation_0']['rmse'][-1]
        else:
            cur_score = evals_log['validation_0']['rmse'][-1]
            if cur_score < self.best_score:
                self.best_model = model
                self.best_score = cur_score
        return False
    
    def after_training(self, model):
        if self.dest_dir is not None:
            if not os.path.exists(self.dest_dir):
                os.makedirs(self.dest_dir)
            dest = f"{self.dest_dir}/seed_{self.seed}.ubj"
            self.best_model.save_model(dest)
            print(f"xgboost model saved at {dest}")
        return model


class SklearnPredictor(Predictor):

    def __init__(self, env, config):
        super().__init__(env, config)
        tmp_config = config._config_dict.copy()
        self.alg_name = tmp_config.pop("alg")
        if self.alg_name == "xgboost":
            self.model = XGBR(**tmp_config)
        elif self.alg_name == "lr":
            self.model = LinearRegression(**tmp_config)
        elif self.alg_name == "dt":
            self.model = DecisionTreeRegressor(**tmp_config)
        else:
            raise ValueError(f"config.alg [{config.alg}] not implemented")
        
        if self.env.save_substitute_model:
            self.checkpoint_dir = f"checkpoint/{env.name}/{self.alg_name}"
        else:
            self.checkpoint_dir = None

    def fit(self, train_params, train_values):
        train_params_np, train_values_np = self.env.dataset_to_numpy(
            train_params, train_values)
        if self.config.alg == "xgboost":
            X_train, X_val, y_train, y_val = train_test_split(
                train_params_np, train_values_np, test_size=self.config.val_ratio)
            cb = MyXgboostTopOneCheckpointCallback(self.checkpoint_dir, self.env.seed)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[cb])
            # overwrite current model with best model
            self.model.get_booster().load_model(cb.best_model.save_raw('ubj'))
        else:
            self.model.fit(train_params_np, train_values_np)
            if self.checkpoint_dir is not None:
                save_sklearn_checkpoint(self.checkpoint_dir, self.alg_name, self.env.seed, self.model)

    def predict(self, test_params):
        test_params_np = self.env.dataset_to_numpy(test_params)
        return self.model.predict(test_params_np)
