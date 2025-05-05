import os
import re
import sys
import subprocess
from pathlib import Path

# Add the project root directory to the Python path
PROJECT_DIRNAME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRNAME))

import scripts.utils as utils


METADATA = {
    "im2latex_formulas.norm.lst": "https://im2markup.yuntiandeng.com/data/im2latex_formulas.norm.lst",
    "im2latex_validate_filter.lst": "https://im2markup.yuntiandeng.com/data/im2latex_validate_filter.lst",
    "im2latex_train_filter.lst": "https://im2markup.yuntiandeng.com/data/im2latex_train_filter.lst",
    "im2latex_test_filter.lst": "https://im2markup.yuntiandeng.com/data/im2latex_test_filter.lst",
    "formula_images.tar.gz": "https://im2markup.yuntiandeng.com/data/formula_images.tar.gz",
}

from config import *

def main():
    DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    cur_dir = os.getcwd()
    os.chdir(DATA_DIRNAME)

    # Download images and grouth truth files
    for filename, url in METADATA.items():
        if not Path(filename).is_file():
            utils.download_url(url, filename)

    # # # Unzip
    if not RAW_IMAGES_DIRNAME.exists():
        RAW_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
        utils.extract_tar_file("formula_images.tar.gz")

    # # Extract regions of interest
    if not PROCESSED_IMAGES_DIRNAME.exists():
        PROCESSED_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
        print("Cropping images...")
        for image_filename in RAW_IMAGES_DIRNAME.glob("*.png"):
            cropped_image = utils.crop(image_filename, padding=8)
            if not cropped_image:
                continue
            cropped_image.save(PROCESSED_IMAGES_DIRNAME / image_filename.name)

    # Clean the ground truth file
    if not CLEANED_FORMULA_FILE.is_file():
        print("Cleaning data...")

        with open(FORMULA_FILE, "r", encoding="utf-8") as infile, open(CLEANED_FORMULA_FILE, "w", encoding="utf-8") as outfile:
            for line in infile:
                line = re.sub(r'\\left\(', '(', line)
                line = re.sub(r'\\right\)', ')', line)
                line = re.sub(r'\\left\[', '[', line)
                line = re.sub(r'\\right\]', ']', line)
                line = re.sub(r'\\left\{', '{', line)
                line = re.sub(r'\\right\}', '}', line)
                line = re.sub(r'\\vspace\s*\{\s*[^}]*\s*\}', '', line)
                line = re.sub(r'\\hspace\s*\{\s*[^}]*\s*\}', '', line)
                outfile.write(line)

    # Build vocabulary
    if not VOCAB_FILE.is_file():
        print("Building vocabulary...")
        all_formulas = utils.get_all_formulas(CLEANED_FORMULA_FILE)
        _, train_formulas = utils.get_split(all_formulas, TRAIN_FILTER_FILE)
        tokenizer = utils.Tokenizer()
        tokenizer.train(train_formulas)
        tokenizer.save(VOCAB_FILE)
    os.chdir(cur_dir)


if __name__ == "__main__":
    main()
