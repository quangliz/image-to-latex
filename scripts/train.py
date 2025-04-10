import sys
import os
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

# Add the project root directory to the Python path
PROJECT_DIRNAME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRNAME))

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from scripts.data import Im2Latex
from model.lit_resnet_transformer import LitResNetTransformer
from scripts.callbacks import MetricsCallback


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):
    datamodule = Im2Latex(**cfg.data)
    datamodule.setup()

    lit_model = LitResNetTransformer(**cfg.lit_model)

    # Ensure checkpoints are saved to the root 'checkpoints' directory
    # regardless of Hydra's working directory
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        checkpoint_config = dict(cfg.callbacks.model_checkpoint)
        checkpoint_config["dirpath"] = str(checkpoint_dir)  # Override dirpath with absolute path
        callbacks.append(ModelCheckpoint(**checkpoint_config))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))

    # Add custom metrics callback
    callbacks.append(MetricsCallback())

    logger: Optional[WandbLogger] = None
    if cfg.logger:
        logger = WandbLogger(**cfg.logger)

    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    if trainer.logger:
        trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)

    # Create a symlink to the latest checkpoint
    try:
        # Find the latest checkpoint in the checkpoints directory
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.ckpt"))
            if checkpoints:
                # Filter out the symlink itself
                checkpoints = [ckpt for ckpt in checkpoints if ckpt.name != "latest.ckpt"]
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    # Create a symlink to the latest checkpoint
                    latest_link = checkpoints_dir / "latest.ckpt"
                    if latest_link.exists():
                        latest_link.unlink()
                    latest_link.symlink_to(latest.name)
                    print(f"Created symlink to latest checkpoint: {latest.name}")
    except Exception as e:
        print(f"Warning: Failed to create symlink to latest checkpoint: {e}")


if __name__ == "__main__":
    main()
