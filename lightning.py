import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.functional import accuracy, auroc
from sklearn.metrics import accuracy_score


class LitUpDown(pl.LightningModule):
    def __init__(self, model, l_rate, **kwargs):
        super().__init__()
        self.model = model
        self.l_rate = l_rate
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, acc, acc_weighted, arearoc = self._shared_eval_step(batch)
        metrics = {"train_acc": acc, 'train_weighted_acc': acc_weighted, "train_auroc": arearoc, 'train_loss': loss}
        self.log_dict(metrics, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, acc_weighted, arearoc = self._shared_eval_step(batch)
        metrics = {"val_acc": acc, "val_auroc": arearoc, 'val_weighted_acc': acc_weighted, 'val_loss': loss}
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc, acc_weighted, arearoc = self._shared_eval_step(batch)
        metrics = {"test_acc": acc, 'test_weighted_acc': acc_weighted, "test_loss": loss, "test_auroc": auroc}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch):
        x, y, ret = batch
        y_hat = self.model(x.float())
        loss = F.binary_cross_entropy(y_hat, y.float().squeeze())
        acc = accuracy(y_hat, y.int())
        acc_weighted = accuracy_score(y.int(), y_hat > 0.5, sample_weight=torch.abs(ret))
        arearoc = auroc(y_hat, y.int(), num_classes=1)

        return loss, acc, acc_weighted, arearoc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, ret = batch
        y_hat = self.model(x.float())
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.l_rate)
