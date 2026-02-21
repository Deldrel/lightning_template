from functools import cache
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger
from pedros import has_dep, get_logger

from app.settings import get_settings


class WandbManager:
    def __init__(self):
        self.logger = get_logger()
        self.settings = get_settings()
        self.enabled = has_dep("wandb") and self.settings.wandb.use_wandb
        self.wandb = __import__("wandb") if self.enabled else None

    def login(self):
        if self.enabled:
            self.wandb.login()

    def init(self):
        if self.enabled:
            return self.wandb.init(
                project=self.settings.wandb.project,
                entity=self.settings.wandb.entity,
                dir=str(self.settings.wandb.root_path),
            )
        return None

    def finish(self):
        if self.enabled:
            self.wandb.finish()

    def save(self, path: Path):
        if self.enabled:
            self.wandb.save(path, base_path=self.settings.wandb.root_path)

    def get_logger(self) -> WandbLogger | None:
        if self.enabled:
            return WandbLogger(
                project=self.settings.wandb.project,
                entity=self.settings.wandb.entity,
                save_dir=str(self.settings.wandb.root_path),
                experiment=self.wandb.run,
            )
        return None


@cache
def get_wandb() -> WandbManager:
    return WandbManager()
