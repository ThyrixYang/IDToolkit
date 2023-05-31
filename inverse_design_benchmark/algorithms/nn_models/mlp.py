import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

class MLP(LightningModule):
    
    def __init__(self, 
                 input_dim,
                 output_dim,
                 config):
        super().__init__()
        if not isinstance(config, dict):
            config = config.to_dict()
        self.config = config
        module_list = []
        module_list.append(nn.Linear(input_dim, config["hidden_size"]))
        for _ in range(config["hidden_num"]):
            module_list.append(nn.ReLU())
            if config["bn"]:
                module_list.append(nn.BatchNorm1d(num_features=config["hidden_size"]))
            module_list.append(nn.Linear(config["hidden_size"], config["hidden_size"]))
        module_list.append(nn.Linear(config["hidden_size"], output_dim))
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