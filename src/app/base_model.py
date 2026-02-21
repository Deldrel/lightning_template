from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch.optim import Adam

from app.settings import AppSettings
from app.wandb_manager import get_wandb


class BaseModel(LightningModule, ABC):
    def __init__(self, settings: AppSettings):
        super(BaseModel, self).__init__()
        self.settings = settings
        self.wandb = get_wandb()
        self.loss_func = self.get_loss_func()
        self.save_hyperparameters(ignore=["settings"])

    @abstractmethod
    def get_loss_func(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.settings.train.learning_rate)
        return optimizer

    def _step(self, batch, loss_name: str) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log(loss_name, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        return self._step(batch, "val_loss")

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        return self._step(batch, "test_loss")

    def on_train_epoch_start(self) -> None:
        self.log(
            "learning_rate",
            self.optimizers().param_groups[0]["lr"],
            prog_bar=True,
            logger=True,
        )

    def on_train_end(self) -> None:
        path = Path("models") / self.__class__.__name__
        path.mkdir(parents=True, exist_ok=True)
        model_name = f"model_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt"
        torch.save(self.state_dict(), path / model_name)
        self.wandb.save(path / model_name)
