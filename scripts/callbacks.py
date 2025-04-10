from pytorch_lightning.callbacks import Callback


class MetricsCallback(Callback):
    """Callback to log metrics after each epoch."""

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log metrics after validation epoch ends."""
        print("Validation epoch ended. Metrics should be logged to wandb.")
        # All metrics are already logged in the LightningModule

    def on_train_epoch_end(self, trainer, pl_module):
        """Log metrics after training epoch ends."""
        print("Training epoch ended. Metrics should be logged to wandb.")
        # All metrics are already logged in the LightningModule
