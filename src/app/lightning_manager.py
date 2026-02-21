from lightning import seed_everything
from pedros import get_logger

from app.data_module import DataModule
from app.mnist_model import MNISTModel
from app.settings import get_settings
from app.trainer import get_trainer
from app.wandb_manager import get_wandb


class LightningManager:
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.wandb = get_wandb()

        self.wandb.login()

    def start_training(self) -> None:
        self.wandb.init()

        seed_everything(self.settings.seed, workers=True)

        data_module = DataModule(self.settings)
        model = MNISTModel(self.settings)
        trainer = get_trainer(self.settings)

        trainer.fit(model, data_module)
        trainer.test(model, data_module)
