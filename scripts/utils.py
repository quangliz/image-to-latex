import json
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py."""

    def update_to(self, blocks=1, bsize=1, tsize=None) -> None:
        """Inform the progress bar how many data have been downloaded.

        Args:
            blocks: Number of blocks transferred so far.
            bsize: Size of each block (in tqdm units).
            tsize: Total size (in tqdm units).
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)


def download_url(url: str, filename: str) -> None:
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        t.set_description(filename)
        urlretrieve(url, filename, reporthook=t.update_to, data=None)


def extract_tar_file(filename: str) -> None:
    """Extract a .tar or .tar.gz file."""
    print(f"Extracting {filename}...")
    with tarfile.open(filename, "r") as f:
        f.extractall()


def pil_loader(fp: Path, mode: str) -> Image.Image:
    with open(fp, "rb") as f:
        try:
            img = Image.open(f)
        except:
            print(f"Error opening {fp}")
            img = Image.fromarray(np.full((64, 128), 255, dtype=np.uint8))
        return img.convert(mode)


def get_all_formulas(filename: Path) -> List[List[str]]:
    """Returns all the formulas in the formula file."""
    with open(filename) as f:
        all_formulas = [formula.strip("\n").split() for formula in f.readlines()]
    return all_formulas


def get_split(
    all_formulas: List[List[str]],
    filename: Path,
) -> Tuple[List[str], List[List[str]]]:
    image_names = []
    formulas = []
    with open(filename) as f:
        for line in f:
            img_name, formula_idx = line.strip("\n").split(" ")
            image_names.append(img_name)
            formulas.append(all_formulas[int(formula_idx)])
    return image_names, formulas


def first_and_last_nonzeros(arr):
    for i in range(len(arr)):
        if arr[i] != 0:
            break
    left = i
    for i in reversed(range(len(arr))):
        if arr[i] != 0:
            break
    right = i
    return left, right


def crop(filename: Path, padding: int = 8) -> Optional[Image.Image]:
    image = pil_loader(filename, mode="RGBA")

    # Replace the transparency layer with a white background
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("L")

    # Invert the color to have a black background and white text
    arr = 255 - np.array(new_image)

    # Area that has text should have nonzero pixel values
    row_sums = np.sum(arr, axis=1)
    col_sums = np.sum(arr, axis=0)
    y_start, y_end = first_and_last_nonzeros(row_sums)
    x_start, x_end = first_and_last_nonzeros(col_sums)

    # Some images have no text
    if y_start >= y_end or x_start >= x_end:
        print(f"{filename.name} is ignored because it does not contain any text")
        return None

    # Cropping
    cropped = arr[y_start : y_end + 1, x_start : x_end + 1]
    H, W = cropped.shape

    # Add paddings
    new_arr = np.zeros((H + padding * 2, W + padding * 2))
    new_arr[padding : H + padding, padding : W + padding] = cropped

    # Invert the color back to have a white background and black text
    new_arr = 255 - new_arr
    return Image.fromarray(new_arr).convert("L")
