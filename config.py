import sys
from pathlib import Path

PROJECT_DIRNAME = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_DIRNAME))

DATA_DIRNAME = PROJECT_DIRNAME / "data"
RAW_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images"
PROCESSED_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images_processed"
VOCAB_FILE = DATA_DIRNAME / "vocab.json"
FORMULA_FILE = DATA_DIRNAME / "im2latex_formulas.norm.lst"
CLEANED_FORMULA_FILE = DATA_DIRNAME / "im2latex_formulas.norm.new.lst"
TRAIN_FILTER_FILE = DATA_DIRNAME / "im2latex_train_filter.lst"
VALIDATE_FILTER_FILE = DATA_DIRNAME / "im2latex_validate_filter.lst"
TEST_FILTER_FILE = DATA_DIRNAME / "im2latex_test_filter.lst"

BEST_CHECKPOINT = PROJECT_DIRNAME / "checkpoints" / "100k" / "epoch_epoch=13_valloss_val" / "loss_epoch=0.21.ckpt"
# print(PROJECT_DIRNAME, DATA_DIRNAME, VOCAB_FILE)

# DATA_DIRNAME = PROJECT_DIRNAME / "tmp"
# RAW_IMAGES_DIRNAME = DATA_DIRNAME / "generated_png_images"
# PROCESSED_IMAGES_DIRNAME = DATA_DIRNAME / "processed_images"
# VOCAB_FILE = DATA_DIRNAME / "230k.json"
# FORMULA_FILE = DATA_DIRNAME / "formulas.lst"
# CLEANED_FORMULA_FILE = DATA_DIRNAME / "formulas.new.lst"
# TRAIN_FILTER_FILE = DATA_DIRNAME / "train.lst"
# VALIDATE_FILTER_FILE = DATA_DIRNAME / "validate.lst"
# TEST_FILTER_FILE = DATA_DIRNAME / "test.lst"