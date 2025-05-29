from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from scripts.data import Tokenizer
from .resnet_transformer import ResNetTransformer
from .metrics import CharacterErrorRate, ExactMatchScore, BLEUScore, EditDistance
from config import VOCAB_FILE

class LitResNetTransformer(LightningModule):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        num_decoder_layers: int,
        max_output_len: int,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        milestones: List[int] = [5],
        gamma: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

        # For tracking epoch-level metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

        vocab_file = VOCAB_FILE
        self.tokenizer = Tokenizer.load(vocab_file)
        self.model = ResNetTransformer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            max_output_len=max_output_len,
            sos_index=self.tokenizer.sos_index,
            eos_index=self.tokenizer.eos_index,
            pad_index=self.tokenizer.pad_index,
            num_classes=len(self.tokenizer),
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index)

        # Training metrics
        self.train_cer = CharacterErrorRate(self.tokenizer.ignore_indices)
        self.train_exact_match = ExactMatchScore(self.tokenizer.ignore_indices)
        self.train_bleu = BLEUScore(self.tokenizer.ignore_indices)
        self.train_edit_distance = EditDistance(self.tokenizer.ignore_indices)

        # Validation metrics
        self.val_cer = CharacterErrorRate(self.tokenizer.ignore_indices)
        self.val_exact_match = ExactMatchScore(self.tokenizer.ignore_indices)
        self.val_bleu = BLEUScore(self.tokenizer.ignore_indices)
        self.val_edit_distance = EditDistance(self.tokenizer.ignore_indices)

        # Test metrics
        self.test_cer = CharacterErrorRate(self.tokenizer.ignore_indices)
        self.test_exact_match = ExactMatchScore(self.tokenizer.ignore_indices)
        self.test_bleu = BLEUScore(self.tokenizer.ignore_indices)
        self.test_edit_distance = EditDistance(self.tokenizer.ignore_indices)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])

        # Log loss for each step (for progress bar and step-level tracking)
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)

        # Store loss for epoch-level logging
        self.training_step_outputs.append(loss.detach())

        # Only calculate metrics on a small subset of batches (1%) to avoid slowing down training
        # This gives us some metrics during training without significant slowdown
        if batch_idx % 100 == 0:  # Calculate metrics every 100 batches
            with torch.no_grad():  # Use no_grad to save memory
                preds = self.model.predict(imgs)
                self.train_cer.update(preds, targets)
                self.train_exact_match.update(preds, targets)
                self.train_bleu.update(preds, targets)
                self.train_edit_distance.update(preds, targets)

        return loss

    def on_train_epoch_end(self):
        # Calculate and log average training loss for the epoch
        if len(self.training_step_outputs) > 0:
            epoch_mean_loss = torch.stack(self.training_step_outputs).mean()
            # Log the epoch average loss
            self.log("train/loss_epoch", epoch_mean_loss, prog_bar=True)
            # Clear the list for the next epoch
            self.training_step_outputs.clear()

        # Calculate and log training metrics at the end of each epoch
        if self.train_cer.total > 0:  # Only log if we have collected some data
            self.log("train/cer", self.train_cer.compute(), prog_bar=True)
            self.log("train/exact_match", self.train_exact_match.compute())
            self.log("train/bleu", self.train_bleu.compute())
            self.log("train/edit_distance", self.train_edit_distance.compute())

            # Reset metrics for next epoch
            self.train_cer.reset()
            self.train_exact_match.reset()
            self.train_bleu.reset()
            self.train_edit_distance.reset()

    def validation_step(self, batch, batch_idx):  # batch_idx is required by PyTorch Lightning
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])

        # Store loss for epoch-level logging
        self.validation_step_outputs.append(loss.detach())

        # Log step-level loss for debugging
        self.log("val/loss_step", loss, on_step=True, on_epoch=False, prog_bar=False)

        with torch.no_grad():  # Use no_grad to save memory
            preds = self.model.predict(imgs)
            # Update metrics (don't log yet, will be logged at epoch end)
            self.val_cer.update(preds, targets)
            self.val_exact_match.update(preds, targets)
            self.val_bleu.update(preds, targets)
            self.val_edit_distance.update(preds, targets)

        return loss

    def on_validation_epoch_end(self):
        # Calculate and log average validation loss for the epoch
        if len(self.validation_step_outputs) > 0:
            epoch_mean_loss = torch.stack(self.validation_step_outputs).mean()
            # Log the epoch average loss with sync_dist=True
            self.log("val/loss_epoch", epoch_mean_loss, prog_bar=True, sync_dist=True)
            # Clear the list for the next epoch
            self.validation_step_outputs.clear()

        # Log computed metrics
        if self.val_cer.total > 0:
            val_cer = self.val_cer.compute()
            val_exact_match = self.val_exact_match.compute()
            val_bleu = self.val_bleu.compute()
            val_edit_distance = self.val_edit_distance.compute()

            self.log("val/cer", val_cer, prog_bar=True, sync_dist=True)
            self.log("val/exact_match", val_exact_match, sync_dist=True)
            self.log("val/bleu", val_bleu, sync_dist=True)
            self.log("val/edit_distance", val_edit_distance, sync_dist=True)

            # Reset metrics for next epoch
            self.val_cer.reset()
            self.val_exact_match.reset()
            self.val_bleu.reset()
            self.val_edit_distance.reset()

            # Print a message to confirm metrics were logged
            print(f"Validation metrics logged: loss_epoch={epoch_mean_loss:.4f}, cer={val_cer:.4f}, exact_match={val_exact_match:.4f}, bleu={val_bleu:.4f}, edit_distance={val_edit_distance:.4f}")

    def test_step(self, batch, batch_idx):  # batch_idx is required by PyTorch Lightning
        imgs, targets = batch
        with torch.no_grad():  # Use no_grad to save memory
            preds = self.model.predict(imgs)
            test_cer = self.test_cer(preds, targets)
            test_exact_match = self.test_exact_match(preds, targets)
            test_bleu = self.test_bleu(preds, targets)
            test_edit_distance = self.test_edit_distance(preds, targets)

            self.log("test/cer", test_cer)
            self.log("test/exact_match", test_exact_match)
            self.log("test/bleu", test_bleu)
            self.log("test/edit_distance", test_edit_distance)
            return preds

    def on_test_epoch_end(self):
        test_outputs = self.trainer.predict_loop.predictions
        with open("test_predictions.txt", "w") as f:
            for preds in test_outputs:
                for pred in preds:
                    decoded = self.tokenizer.decode(pred.tolist())
                    decoded.append("\n")
                    decoded_str = " ".join(decoded)
                    f.write(decoded_str)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]
