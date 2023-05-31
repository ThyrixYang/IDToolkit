import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .mlp import MLP

class ReverseNetwork(LightningModule):
    
    def __init__(self,
                 x_dim,
                 y_dim,
                 config):
        super().__init__()
        self.config = config
        self.model = MLP(input_dim=y_dim, 
                           output_dim=x_dim,
                           config=config)
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return opt
        
    def _shared_step(self, x, y):
        pred_x = self.model(y)
        loss = torch.mean((x - pred_x)**2)
        return loss
    
    def training_step(self, batch, batch_index):
        x, y = batch
        loss = self._shared_step(x, y)
        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        loss = self._shared_step(x, y)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_index):
        y = batch[0]
        pred_x = self.model(y)
        return pred_x