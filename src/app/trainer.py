from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pedros import has_dep

from app.settings import AppSettings
from app.wandb_manager import get_wandb


def get_trainer(settings: AppSettings) -> Trainer:
    wandb = get_wandb()

    callbacks = []

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=settings.train.callback_verbose,
    )
    callbacks.append(early_stopping)

    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="mnist-{epoch:02d}-{val_loss:.2f}",
        verbose=settings.train.callback_verbose,
    )
    callbacks.append(model_checkpoint)

    if has_dep("rich"):
        callbacks.append(RichProgressBar())

    return Trainer(
        accelerator="cpu" if settings.train.force_cpu else "auto",
        max_epochs=settings.train.max_epochs,
        callbacks=callbacks,
        logger=wandb.get_logger(),
        fast_dev_run=settings.train.fast_dev_run,
        enable_progress_bar=True,
    )
