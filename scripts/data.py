import json
import random
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from pytorch_lightning import LightningDataModule


from scripts.utils import pil_loader, get_all_formulas, get_split

class BaseDataset(Dataset):
    """A base Dataset class.

    Args:
        image_filenames: (N, *) feature vector.
        targets: (N, *) target vector relative to data.
        transform: Feature transformation.
        target_transform: Target transformation.
    """

    def __init__(
        self,
        root_dir: Path,
        image_filenames: List[str],
        formulas: List[List[str]],
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        assert len(image_filenames) == len(formulas)
        self.root_dir = root_dir
        self.image_filenames = image_filenames
        self.formulas = formulas
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.formulas)

    def __getitem__(self, idx: int):
        """Returns a sample from the dataset at the given index."""
        image_filename, formula = self.image_filenames[idx], self.formulas[idx]
        image_filepath = self.root_dir / image_filename
        if image_filepath.is_file():
            image = pil_loader(image_filepath, mode="L")
        else:
            # Returns a blank image if cannot find the image
            image = Image.fromarray(np.full((64, 128), 255, dtype=np.uint8))
            formula = []
        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"]
        return image, formula

class Tokenizer:
    def __init__(self, token_to_index: Optional[Dict[str, int]] = None) -> None:
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.token_to_index: Dict[str, int]
        self.index_to_token: Dict[int, str]

        if token_to_index:
            self.token_to_index = token_to_index
            self.index_to_token = {index: token for token, index in self.token_to_index.items()}
            self.pad_index = self.token_to_index[self.pad_token]
            self.sos_index = self.token_to_index[self.sos_token]
            self.eos_index = self.token_to_index[self.eos_token]
            self.unk_index = self.token_to_index[self.unk_token]
        else:
            self.token_to_index = {}
            self.index_to_token = {}
            self.pad_index = self._add_token(self.pad_token)
            self.sos_index = self._add_token(self.sos_token)
            self.eos_index = self._add_token(self.eos_token)
            self.unk_index = self._add_token(self.unk_token)

        self.ignore_indices = {self.pad_index, self.sos_index, self.eos_index, self.unk_index}

    def _add_token(self, token: str) -> int:
        """Add one token to the vocabulary.

        Args:
            token: The token to be added.

        Returns:
            The index of the input token.
        """
        if token in self.token_to_index:
            return self.token_to_index[token]
        index = len(self)
        self.token_to_index[token] = index
        self.index_to_token[index] = token
        return index

    def __len__(self):
        return len(self.token_to_index)

    def train(self, formulas: List[List[str]], min_count: int = 2) -> None:
        """Create a mapping from tokens to indices and vice versa.

        Args:
            formulas: Lists of tokens.
            min_count: Tokens that appear fewer than `min_count` will not be
                included in the mapping.
        """
        # Count the frequency of each token
        counter: Dict[str, int] = {}
        for formula in formulas:
            for token in formula:
                counter[token] = counter.get(token, 0) + 1

        for token, count in counter.items():
            # Remove tokens that show up fewer than `min_count` times
            if count < min_count:
                continue
            index = len(self)
            self.index_to_token[index] = token
            self.token_to_index[token] = index

    def encode(self, formula: List[str]) -> List[int]:
        indices = [self.sos_index]
        for token in formula:
            index = self.token_to_index.get(token, self.unk_index)
            indices.append(index)
        indices.append(self.eos_index)
        return indices

    def decode(self, indices: List[int], inference: bool = True) -> List[str]:
        tokens = []
        for index in indices:
            if index not in self.index_to_token:
                raise RuntimeError(f"Found an unknown index {index}")
            if index == self.eos_index:
                break
            if inference and index in self.ignore_indices:
                continue
            token = self.index_to_token[index]
            tokens.append(token)
        return tokens

    def save(self, filename: Union[Path, str]):
        """Save token-to-index mapping to a json file."""
        with open(filename, "w") as f:
            json.dump(self.token_to_index, f)

    @classmethod
    def load(cls, filename: Union[Path, str]) -> "Tokenizer":
        """Create a `Tokenizer` from a mapping file outputted by `save`.

        Args:
            filename: Path to the file to read from.

        Returns:
            A `Tokenizer` object.
        """
        with open(filename) as f:
            token_to_index = json.load(f)
        return cls(token_to_index)


class Im2Latex(LightningDataModule):
    """Data processing for the Im2Latex-100K dataset.

    Args:
        batch_size: The number of samples per batch.
        num_workers: The number of subprocesses to use for data loading.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
            before returning them.
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_dirname = Path(__file__).resolve().parents[1] / "data"
        self.vocab_file = Path(__file__).resolve().parents[1] / "data" / "vocab.json"
        formula_file = self.data_dirname / "im2latex_formulas.norm.new.lst"
        if not formula_file.is_file():
            raise FileNotFoundError("Did you run scripts/prepare_data.py?")
        self.all_formulas = get_all_formulas(formula_file)
        self.transform = {
            "train": A.Compose(
                [
                    A.Affine(scale=(0.6, 1.0), rotate=(-1, 1), p=0.5),
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(blur_limit=(1, 1), p=0.5),
                    ToTensorV2(),
                ]
            ),
            "val/test": ToTensorV2(),
        }

    @property
    def processed_images_dirname(self):
        return self.data_dirname / "formula_images_processed"

    def setup(self, stage: Optional[str] = None) -> None:
        """Load images and formulas, and assign them to a `torch Dataset`.

        `self.train_dataset`, `self.val_dataset` and `self.test_dataset` will
        be assigned after this method is called.
        """
        self.tokenizer = Tokenizer.load(self.vocab_file)

        if stage in ("fit", None):
            train_image_names, train_formulas = get_split(
                self.all_formulas,
                self.data_dirname / "im2latex_train_filter.lst",
            )
            self.train_dataset = BaseDataset(
                self.processed_images_dirname,
                image_filenames=train_image_names,
                formulas=train_formulas,
                transform=self.transform["train"],
            )

            val_image_names, val_formulas = get_split(
                self.all_formulas,
                self.data_dirname / "im2latex_validate_filter.lst",
            )
            self.val_dataset = BaseDataset(
                self.processed_images_dirname,
                image_filenames=val_image_names,
                formulas=val_formulas,
                transform=self.transform["val/test"],
            )

        if stage in ("test", None):
            test_image_names, test_formulas = get_split(
                self.all_formulas,
                self.data_dirname / "im2latex_test_filter.lst",
            )
            self.test_dataset = BaseDataset(
                self.processed_images_dirname,
                image_filenames=test_image_names,
                formulas=test_formulas,
                transform=self.transform["val/test"],
            )

    def collate_fn(self, batch):
        images, formulas = zip(*batch)
        B = len(images)
        max_H = max(image.shape[1] for image in images)
        max_W = max(image.shape[2] for image in images)
        max_length = max(len(formula) for formula in formulas)
        padded_images = torch.zeros((B, 1, max_H, max_W))
        batched_indices = torch.zeros((B, max_length + 2), dtype=torch.long)
        for i in range(B):
            H, W = images[i].shape[1], images[i].shape[2]
            y, x = random.randint(0, max_H - H), random.randint(0, max_W - W)
            padded_images[i, :, y : y + H, x : x + W] = images[i]
            indices = self.tokenizer.encode(formulas[i])
            batched_indices[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)
        return padded_images, batched_indices

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
