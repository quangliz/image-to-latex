from typing import List, Set

import editdistance
import torch
from torch import Tensor
from torchmetrics import Metric
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class CharacterErrorRate(Metric):
    def __init__(self, ignore_indices: Set[int], *args):
        super().__init__(*args)
        self.ignore_indices = ignore_indices
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.error: Tensor
        self.total: Tensor

    def update(self, preds, targets):
        N = preds.shape[0]
        for i in range(N):
            pred = [token for token in preds[i].tolist() if token not in self.ignore_indices]
            target = [token for token in targets[i].tolist() if token not in self.ignore_indices]
            distance = editdistance.distance(pred, target)
            if max(len(pred), len(target)) > 0:
                self.error += distance / max(len(pred), len(target))
        self.total += N

    def compute(self) -> Tensor:
        return self.error / self.total


class ExactMatchScore(Metric):
    def __init__(self, ignore_indices: Set[int], *args):
        super().__init__(*args)
        self.ignore_indices = ignore_indices
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.correct: Tensor
        self.total: Tensor

    def update(self, preds, targets):
        N = preds.shape[0]
        for i in range(N):
            pred = [token for token in preds[i].tolist() if token not in self.ignore_indices]
            target = [token for token in targets[i].tolist() if token not in self.ignore_indices]
            if pred == target:
                self.correct += 1
        self.total += N

    def compute(self) -> Tensor:
        return self.correct / self.total


class BLEUScore(Metric):
    def __init__(self, ignore_indices: Set[int], *args):
        super().__init__(*args)
        self.ignore_indices = ignore_indices
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.score: Tensor
        self.total: Tensor
        self.smoothing_function = SmoothingFunction().method1

    def update(self, preds, targets):
        N = preds.shape[0]
        for i in range(N):
            pred = [token for token in preds[i].tolist() if token not in self.ignore_indices]
            target = [token for token in targets[i].tolist() if token not in self.ignore_indices]

            # Convert token IDs to strings for BLEU calculation
            pred_str = [str(token) for token in pred]
            target_str = [str(token) for token in target]

            # Calculate BLEU score (using smoothing to handle edge cases)
            if len(pred_str) > 0 and len(target_str) > 0:
                bleu = sentence_bleu([target_str], pred_str, smoothing_function=self.smoothing_function)
                self.score += torch.tensor(bleu)
        self.total += N

    def compute(self) -> Tensor:
        return self.score / self.total


class EditDistance(Metric):
    def __init__(self, ignore_indices: Set[int], *args):
        super().__init__(*args)
        self.ignore_indices = ignore_indices
        self.add_state("distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.distance: Tensor
        self.total: Tensor

    def update(self, preds, targets):
        N = preds.shape[0]
        for i in range(N):
            pred = [token for token in preds[i].tolist() if token not in self.ignore_indices]
            target = [token for token in targets[i].tolist() if token not in self.ignore_indices]

            # Calculate raw edit distance
            distance = editdistance.distance(pred, target)
            self.distance += distance
        self.total += N

    def compute(self) -> Tensor:
        return self.distance / self.total
