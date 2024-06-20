from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class WandbConfig(BaseModel):
    # api_key is handled in .env file
    entity: str = 'deldrel'
    project: str = 'lightning_template'


class DataModuleConfig(BaseModel):
    data_dir: str = 'data'
    batch_size: int = 32
    num_workers: int = 8
    persistent_workers: bool = True
    training_set_ratio: float = 0.8


class ModelConfig(BaseModel):
    architecture: str = 'MLP'
    learning_rate: float = 0.01
    loss_function: str = 'CrossEntropyLoss'
    optimizer: str = 'NAdam'


class TrainerConfig(BaseModel):
    max_epochs: int = 50
    save_dir: str = 'logs/models'


class CheckpointConfig(BaseModel):
    dirpath: str = 'logs/checkpoints'
    filename: str = 'checkpoint-{epoch:02d}-{val_loss:.6f}'
    monitor: str = 'val_loss'
    save_last: bool = True
    save_top_k: int = 1
    mode: str = 'min'


class EarlyStoppingConfig(BaseModel):
    monitor: str = 'val_loss'
    min_delta: float = 0.001
    patience: int = 10  # 10
    mode: str = 'min'


class ReduceLROnPlateauConfig(BaseModel):
    monitor: str = 'val_loss'
    factor: float = 0.1
    patience: int = 5  # 5
    mode: str = 'min'


class Config(BaseModel):
    seed: int = 42
    logdir: str = 'logs'
    verbose: bool = True

    wandb: WandbConfig = WandbConfig()
    data_module: DataModuleConfig = DataModuleConfig()
    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    reduce_lr_on_plateau: ReduceLROnPlateauConfig = ReduceLROnPlateauConfig()

    def dump(self):
        return self.model_dump()


@lru_cache(maxsize=1)
def get_config() -> Config:
    try:
        if not load_dotenv(verbose=True):
            raise ValidationError('No .env variable set')
        return Config()
    except ValidationError as e:
        print(e)
        exit(1)


config: Config = get_config()