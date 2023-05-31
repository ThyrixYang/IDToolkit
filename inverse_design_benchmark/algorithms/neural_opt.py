import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .opt_base import Algorithm
from .nn_models import CVAE, CGAN, GradientDescent, Tandem, ReverseNetwork


class NeuralOptAlgorithm(Algorithm):

    def __init__(self,
                 env,
                 config):
        super().__init__(env=env, config=config)
        if config.net == "cvae":
            self.model = CVAE(x_dim=env.get_input_dim,
                              y_dim=env.get_output_dim,
                              config=config)
        elif config.net == "cgan":
            self.model = CGAN(x_dim=env.get_input_dim,
                              y_dim=env.get_output_dim,
                              config=config)
        elif config.net == "gd":
            self.model = GradientDescent(x_dim=env.get_input_dim,
                                         y_dim=env.get_output_dim,
                                         config=config)
        elif config.net == "reverse_model":
            self.model = ReverseNetwork(x_dim=env.get_input_dim,
                                        y_dim=env.get_output_dim,
                                        config=config)
        elif config.net == "tandem":
            self.model = Tandem(x_dim=env.get_input_dim,
                                y_dim=env.get_output_dim,
                                config=config)
        else:
            raise ValueError(f"config.net [{config.net}] not implemented")

        # saves top-K checkpoints based on "val_loss" metric
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            verbose=True
        )
        
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=config.max_epochs,
            callbacks=[checkpoint_callback],
            inference_mode=config.net not in ["gd"],
        )

    def fit(self, train_params, train_values):
        train_params_np, train_values_np = self.env.dataset_to_numpy(
            train_params, train_values)
        x, y = torch.tensor(train_params_np).float(
        ), torch.tensor(train_values_np).float()
        self.train_x_mean_std = [x.mean(dim=0), x.std(dim=0)]
        self.train_y_mean_std = [y.mean(dim=0), y.std(dim=0)]

        x = (x - self.train_x_mean_std[0]) / (self.train_x_mean_std[1] + 1e-8)
        y = (y - self.train_y_mean_std[0]) / (self.train_y_mean_std[1] + 1e-8)

        dataset = TensorDataset(x, y)
        valid_set_size = int(len(dataset) * self.config.val_ratio)
        train_set_size = len(dataset) - valid_set_size
        train_set, valid_set = torch.utils.data.random_split(
            dataset, [train_set_size, valid_set_size])
        train_loader = DataLoader(train_set,
                                  batch_size=self.config.batch_size,
                                  num_workers=self.config.num_workers)
        val_loader = DataLoader(valid_set,
                                batch_size=self.config.batch_size,
                                num_workers=self.config.num_workers)
        self.trainer.fit(model=self.model,
                         train_dataloaders=train_loader,
                         val_dataloaders=val_loader)

    def search(self, target=None, num_samples=1):
        if target is None:
            target = self.env.target
        if self.config.net in ["reverse_model", "tandem"]:
            # Deterministic methods always output same value, 
            # so we only need to predict once
            num_samples = 1
            
        target = torch.from_numpy(target.copy()).view((1, -1)).float()
        target = (target.repeat(num_samples, 1) -
                  self.train_y_mean_std[0]) / (self.train_y_mean_std[1] + 1e-8)
        test_dataset = TensorDataset(target)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.config.batch_size,
                                 num_workers=self.config.num_workers)
        pred_params = self.trainer.predict(
            self.model, test_loader, ckpt_path="best")
        pred_params = torch.cat(pred_params, dim=0).cpu()
        pred_params = (
            pred_params * self.train_x_mean_std[1] + self.train_x_mean_std[0]).numpy()
        params = [self.env.parameter_space.from_numpy(p) for p in pred_params]
        return params
