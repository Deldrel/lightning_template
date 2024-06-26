from functools import lru_cache
from pathlib import Path

import torch
import wandb
import yaml

from src.config import config
from src.helpers.decorators import timer
from .data_module import DataModule
from .mlp import MLP
from .trainer import get_trainer


class LightningManager:
    def __init__(self):
        self.data_module = None
        self.model = None
        self.trainer = None

    def setup(self) -> None:
        self.data_module = DataModule()

        if config.model.architecture == 'MLP':
            self.model = MLP()
        else:
            raise ValueError(f"Unknown architecture: {config.model.architecture}")

        # self.search_checkpoint()
        self.trainer = get_trainer()

    def search_checkpoint(self) -> None:
        path = Path(config.trainer.save_dir)
        if not path.exists():
            return
        checkpoints = sorted(path.glob('*.pt'))
        if not checkpoints:
            return

        if input(f"Load {checkpoints[-1]}? [y/n]: ") == "y":
            try:
                checkpoint = torch.load(checkpoints[-1])
                self.model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Error loading checkpoint: {e}, keeping new model.")

    @timer
    def train_model(self) -> None:
        self.setup()

        try:
            wandb.init(project=config.wandb.project,
                       entity=config.wandb.entity,
                       dir=config.logdir,
                       config=config.dump())

            print(f"NOTE: you can interrupt the training whenever you want with a keyboard interrupt (CTRL+C)")
            self.trainer.fit(self.model, self.data_module)
            self.trainer.test(self.model, self.data_module)

        except Exception as e:
            if config.verbose:
                print(f"Error training model: {e}")

        finally:
            wandb.finish()

    def sweep_train(self):
        wandb.init(dir=config.logdir,
                   config=config.dump())
        config.update_from_dict(wandb.config)
        self.setup()
        self.trainer.fit(self.model, self.data_module)
        wandb.finish()

    @timer
    def start_sweep(self) -> None:
        with open(config.wandb.sweep_config, 'r') as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config,
                               project=config.wandb.project,
                               entity=config.wandb.entity)
        wandb.agent(sweep_id, function=self.sweep_train)


@lru_cache(maxsize=1)
def get_lightning_manager() -> LightningManager:
    return LightningManager()


lightning_manager = get_lightning_manager()
