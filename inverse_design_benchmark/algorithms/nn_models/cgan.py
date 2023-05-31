import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .mlp import MLP

class CGAN(LightningModule):
    
    def __init__(self,
                 x_dim,
                 y_dim,
                 config):
        super().__init__()
        self.config = config
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.generator = MLP(input_dim=config.z_dim+y_dim, 
                           output_dim=x_dim,
                           config=config)
        self.discriminator = MLP(input_dim=x_dim+y_dim, 
                           output_dim=1,
                           config=config)
        
    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=self.config.lr)
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.lr)
        return [opt_G, opt_D], []
    
    def sample_z(self, n):
        sample = torch.randn(n, self.config.z_dim, device=self.device)
        return sample

    def sample_G(self, n, y):
        z = self.sample_z(n)
        return self.generator(torch.cat([z, y], dim=1))
    
    def training_step(self, batch, batch_index, optimizer_idx):
        x, y = batch
        batch_size = x.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.sample_G(batch_size, y)

        ######################
        # Optimize Generator #
        ######################
        if optimizer_idx == 0:
            d_z = self.discriminator(torch.cat([g_X, y], dim=1))
            # return torch.mean(d_z) * 0.0
            self.log("g_acc", torch.mean((d_z > 0).float()), prog_bar=True)
            errG = F.binary_cross_entropy_with_logits(d_z, real_label)
            self.log("g_loss", errG, prog_bar=True)
            return errG
        ##########################
        # Optimize Discriminator #
        ##########################
        elif optimizer_idx == 1:
            d_x = self.discriminator(torch.cat([x, y], dim=1))
            # return torch.mean(d_x) * 0.0
            errD_real = F.binary_cross_entropy_with_logits(d_x, real_label)
            self.log("d_acc_real", torch.mean((d_x > 0).float()), prog_bar=True)

            d_z = self.discriminator(torch.cat([g_X.detach(), y], dim=1))
            errD_fake = F.binary_cross_entropy_with_logits(d_z, fake_label)
            self.log("d_acc_fake", torch.mean((d_z < 0).float()), prog_bar=True)

            errD = errD_real + errD_fake
            self.log("d_loss", errD, prog_bar=True)
            return errD
        else:
            raise ValueError()
        
    def validation_step(self, batch, batch_idx):
        # the training loss of GAN is meaningless, so we use the latest model
        self.log("val_loss", -self.current_epoch)
        return self.current_epoch
    
    def predict_step(self, batch, batch_index):
        y = batch[0]
        batch_size = y.shape[0]
        z_sample = self.sample_z(batch_size)
        pred_x = self.generator(torch.cat([z_sample, y], dim=1))       
        return pred_x