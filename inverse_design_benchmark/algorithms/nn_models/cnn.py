from typing import Any
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

from .mlp import MLP

class CNN(LightningModule):
    
    def __init__(self, 
                 input_dim,
                 output_dim,
                 config):
        super().__init__()
        if not isinstance(config, dict):
            config = config.to_dict()
        self.config = config
        self.save_hyperparameters('input_dim', 'output_dim')
        self.save_hyperparameters({'config': self.config})
        module_list = [] # N, input_dim
        module_list.append(MLP(input_dim, output_dim, config)) # N, output_dim
        module_list.append(nn.Unflatten(1, (1, -1))) # N, 1, output_dim
        module_list.append(nn.ConvTranspose1d(1, self.config["hidden_size"], 5, 1, 2)) # N, hidden_size, output_dim
        if self.config["bn"]:
            module_list.append(nn.BatchNorm1d(num_features=self.config["hidden_size"]))
        module_list.append(nn.ReLU())
        module_list.append(nn.ConvTranspose1d(self.config["hidden_size"], self.config["hidden_size"], 5, 1, 2))
        module_list.append(nn.ReLU())
        module_list.append(nn.ConvTranspose1d(self.config["hidden_size"], 1, 5, 1, 2)) # N, 1, output_dim
        module_list.append(nn.Flatten(1, -1)) # N, output_dim
        self.model = nn.Sequential(*module_list)
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return opt
    
    def _shared_step(self, x, y):
        pred_y = self.model(x)
        loss = torch.mean((pred_y - y)**2)
        return loss
    
    def training_step(self, batch, batch_index):
        x, y = batch
        loss = self._shared_step(x, y)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        loss = self._shared_step(x, y)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss
        
    def predict_step(self, batch, batch_idx):
        x = batch[0]
        pred_y = self.model(x)
        return pred_y