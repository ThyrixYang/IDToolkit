import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .predictor_base import Predictor
from .nn_models import MLP, CNN
from ..utils.state import save_std_and_mean

class NeuralNetworkPredictor(Predictor):
    
    def __init__(self, env, config):
        super().__init__(env, config)
        if config.net == "mlp":
            self.model = MLP(input_dim=env.get_input_dim, 
                             output_dim=env.get_output_dim,
                             config=config)
        elif config.net == "cnn":
            self.model = CNN(input_dim=env.get_input_dim, 
                             output_dim=env.get_output_dim,
                             config=config)
        else:
            raise ValueError(f"config.net [{config.net}] not implemented")

        # saves top-K checkpoints based on "val_loss" metric
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        # save the models trained under checkpoint directory for future use as substitute models
        if self.env.save_substitute_model:
            self.checkpoint_env_dir = f'checkpoint/{env.name}'
            logger = TensorBoardLogger(
                save_dir=f'{self.checkpoint_env_dir}/{config.net}',
                name=f'seed_{self.env.seed}')
        else:
            self.checkpoint_env_dir = None
            logger = True

        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=config.max_epochs,
            callbacks=[checkpoint_callback],
            logger=logger
        )
        
    def fit(self, train_params, train_values):
        train_params_np, train_values_np = self.env.dataset_to_numpy(train_params, train_values)
        x, y = torch.tensor(train_params_np).float(), torch.tensor(train_values_np).float()
        self.train_x_mean_std = [x.mean(dim=0), x.std(dim=0)]
        
        # data normalization is expected for deep models
        #     if the checkpoint is used as substitue model in inverse design problems, their inputs should also be normalized 
        #     save the std and mean of the train data
        if self.checkpoint_env_dir is not None:
            save_std_and_mean(self.train_x_mean_std[0], self.train_x_mean_std[1], self.checkpoint_env_dir)

        x = (x - self.train_x_mean_std[0]) / (self.train_x_mean_std[1] + 1e-8)
        
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
    
    def predict(self, test_params):
        test_params_np = self.env.dataset_to_numpy(test_params)
        x = torch.tensor(test_params_np).float()
        test_x = (x - self.train_x_mean_std[0]) / (self.train_x_mean_std[1] + 1e-8)
        test_dataset = TensorDataset(test_x)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=self.config.batch_size, 
                                 num_workers=self.config.num_workers)
        pred_values = self.trainer.predict(self.model, test_loader, ckpt_path="best")
        pred_values = torch.cat(pred_values, dim=0)
        return pred_values.cpu().numpy()