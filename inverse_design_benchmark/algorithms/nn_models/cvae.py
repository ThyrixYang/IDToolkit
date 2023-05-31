import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .mlp import MLP

class CVAE(LightningModule):
    
    def __init__(self,
                 x_dim,
                 y_dim,
                 config):
        super().__init__()
        self.config = config
        self.encoder = MLP(input_dim=x_dim+y_dim, 
                           output_dim=config.z_dim*2,
                           config=config)
        self.decoder = MLP(input_dim=config.z_dim+y_dim, 
                           output_dim=x_dim,
                           config=config)
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return opt
        
    def _shared_step(self, x, y):
        encoder_input = torch.cat([x, y], dim=1)
        z_out = self.encoder(encoder_input)
        z_mean, z_logvar = torch.chunk(z_out, 2, dim=1)
        z_sample = torch.exp(0.5*z_logvar) * \
            torch.randn_like(z_logvar) + z_mean
        decoder_input = torch.cat([z_sample, y], dim=1)
        _x = self.decoder(decoder_input)
        kl_loss = -0.5 * \
            torch.mean(1 + z_logvar - torch.pow(z_mean, 2) -
                        torch.exp(z_logvar), dim=-1)
        mse_loss = torch.mean((x - _x)**2, dim=-1)
        loss = torch.mean(kl_loss*self.config.alpha + mse_loss)
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
        batch_size = y.shape[0]
        z_sample = torch.randn(size=(batch_size, self.config.z_dim), device=y.device).float()
        decoder_input = torch.cat([z_sample, y], dim=1)
        pred_x = self.decoder(decoder_input)
        return pred_x