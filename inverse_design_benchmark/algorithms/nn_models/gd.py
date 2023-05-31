import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .mlp import MLP


class GradientDescent(LightningModule):

    def __init__(self,
                 x_dim,
                 y_dim,
                 config):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.config = config
        self.model = MLP(input_dim=x_dim,
                         output_dim=y_dim,
                         config=config)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return opt

    def _shared_step(self, x, y):
        pred_y = self.model(x)
        loss = torch.mean((y - pred_y)**2)
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
        
        # We have normalized x to have mean=0, std=1, so we can sample from N(0, 1) here
        x_sample = torch.randn(size=(batch_size * self.config.pred_repeat_num, 
                                    self.x_dim), 
                            device=y.device,
                            requires_grad=True,
                            dtype=y.dtype).float()
        optimizer = torch.optim.Adam([x_sample], lr=self.config.gd_lr)
        lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=30, threshold=1e-4, verbose=False)
        self.model.eval()
        with torch.enable_grad():
            for _step in range(self.config.gd_step):
                y_pred = self.model(x_sample)
                loss = torch.mean((y_pred.view((batch_size, self.config.pred_repeat_num, -1))
                                - y.unsqueeze(1)) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_schedular.step(loss.item())
        
        final_y_pred = self.model(x_sample)
        final_y_loss = torch.mean((final_y_pred.view((batch_size, self.config.pred_repeat_num, -1))
                                   - y.unsqueeze(1))**2, dim=2)
        eval_cut = max(1, int(self.config.eval_cut * self.config.pred_repeat_num))
        use_idx = final_y_loss.cpu().argsort(dim=1)[:, eval_cut]
        final_pred_x = x_sample.detach().cpu().view((batch_size, self.config.pred_repeat_num, -1)
                                                    )[list(range(batch_size)), use_idx]
        final_pred_x = final_pred_x
        return final_pred_x
