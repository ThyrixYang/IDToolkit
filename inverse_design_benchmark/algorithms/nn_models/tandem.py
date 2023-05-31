import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .mlp import MLP

class Tandem(LightningModule):
    
    def __init__(self,
                 x_dim,
                 y_dim,
                 config):
        super().__init__()
        self.config = config
        self.forward_model = MLP(input_dim=x_dim, 
                           output_dim=y_dim,
                           config=config)
        self.reverse_model = MLP(input_dim=y_dim,
                                 output_dim=x_dim,
                                 config=config)
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return opt
        
    def _shared_step(self, x, y):
        pred_y = self.forward_model(x)
        loss_forward = torch.mean((y - pred_y)**2)
        
        pred_x = self.reverse_model(y)
        _pred_y = self.forward_model(pred_x)
        loss_reverse_forward = torch.mean((y - _pred_y)**2)
        loss = loss_forward + loss_reverse_forward
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
        pred_x = self.reverse_model(y)
        return pred_x