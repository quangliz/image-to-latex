#!/usr/bin/env python
"""
Check if the environment is set up correctly for the img2latex project.
This script verifies that all required dependencies are installed and the data directory is set up correctly.
"""

import sys
import os
from pathlib import Path
import importlib
import subprocess

# Required packages
REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "pytorch_lightning",
    "albumentations",
    "hydra",
    "omegaconf",
    "wandb",
    "PIL",
    "numpy",
    "tqdm",
]

# Required data files
REQUIRED_DATA_FILES = [
    "data/im2latex_formulas.norm.new.lst",
    "data/im2latex_train_filter.lst",
    "data/im2latex_validate_filter.lst",
    "data/im2latex_test_filter.lst",
    "data/vocab.json",
]

# Required directories
REQUIRED_DIRECTORIES = [
    "data/formula_images_processed",
]


def check_packages():
    """Check if all required packages are installed."""
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is NOT installed")
    
    return missing_packages


def check_data_files():
    """Check if all required data files exist."""
    missing_files = []
    for file_path in REQUIRED_DATA_FILES:
        if not Path(file_path).is_file():
            missing_files.append(file_path)
            print(f"❌ {file_path} does NOT exist")
        else:
            print(f"✅ {file_path} exists")
    
    return missing_files


def check_directories():
    """Check if all required directories exist."""
    missing_dirs = []
    for dir_path in REQUIRED_DIRECTORIES:
        if not Path(dir_path).is_dir():
            missing_dirs.append(dir_path)
            print(f"❌ {dir_path} does NOT exist")
        else:
            print(f"✅ {dir_path} exists")
    
    return missing_dirs


def main():
    """Main function to check the environment."""
    print("Checking environment for img2latex project...\n")
    
    # Check packages
    print("Checking required packages:")
    missing_packages = check_packages()
    
    # Check data files
    print("\nChecking required data files:")
    missing_files = check_data_files()
    
    # Check directories
    print("\nChecking required directories:")
    missing_dirs = check_directories()
    
    # Print summary
    print("\nEnvironment check summary:")
    if not missing_packages and not missing_files and not missing_dirs:
        print("✅ All checks passed! The environment is set up correctly.")
    else:
        print("❌ Some checks failed. Please fix the issues below:")
        
        if missing_packages:
            print("\nMissing packages:")
            print("Run the following command to install them:")
            print(f"pip install {' '.join(missing_packages)}")
        
        if missing_files or missing_dirs:
            print("\nMissing data files or directories:")
            print("Run the following command to prepare the data:")
            print("python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
