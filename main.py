#!/usr/bin/env python
"""
Main entry point for the image-to-latex project.
This script provides a command-line interface for training, testing, and inference.
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess

# Add the project root directory to the Python path
PROJECT_DIRNAME = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIRNAME))

def main():
    # Import Path here to ensure it's available in the function scope
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Image-to-LaTeX CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="config.yaml",
                             help="Path to the config file")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test the model")
    test_parser.add_argument("--checkpoint", type=str, required=False, default="checkpoints/latest.ckpt",
                            help="Path to the model checkpoint (default: checkpoints/latest.ckpt)")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference on an image")
    infer_parser.add_argument("--image", type=str, required=True,
                             help="Path to the input image")
    infer_parser.add_argument("--checkpoint", type=str, required=False, default="checkpoints/latest.ckpt",
                             help="Path to the model checkpoint (default: checkpoints/latest.ckpt)")

    args = parser.parse_args()

    if args.command == "train":
        # Check if data files exist, if not run prepare_data.py
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        formula_file = data_dir / "im2latex_formulas.norm.new.lst"
        vocab_file = data_dir / "vocab.json"

        if not formula_file.is_file() or not vocab_file.is_file():
            print("Data files not found. Running data preparation script...")
            prepare_cmd = [sys.executable, "scripts/prepare_data.py"]
            print(f"Running: {' '.join(prepare_cmd)}")
            try:
                subprocess.run(prepare_cmd, check=True)
                print("Data preparation completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error during data preparation: {e}")
                sys.exit(1)

        # For training, we'll directly call the train.py script
        # This ensures Hydra is initialized correctly
        cmd = [sys.executable, "scripts/train.py"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # After training, create a symlink to the latest checkpoint
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

    elif args.command == "test":
        import torch
        from model.lit_resnet_transformer import LitResNetTransformer
        from pytorch_lightning import Trainer

        # Determine device (CPU or GPU)
        device = "gpu" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Check if checkpoint exists
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
            print("Please provide a valid checkpoint file with --checkpoint.")
            sys.exit(1)

        try:
            model = LitResNetTransformer.load_from_checkpoint(args.checkpoint)
            trainer = Trainer(accelerator=device)
            trainer.test(model)
        except Exception as e:
            print(f"Error during testing: {e}")
            sys.exit(1)

    elif args.command == "infer":
        import torch
        import numpy as np
        from pathlib import Path
        from albumentations.pytorch.transforms import ToTensorV2
        from model.lit_resnet_transformer import LitResNetTransformer

        # Determine device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Check if checkpoint exists
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
            print("Please provide a valid checkpoint file with --checkpoint.")
            sys.exit(1)

        # Check if image exists
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found.")
            print("Please provide a valid image file with --image.")
            sys.exit(1)

        try:
            # Load the model
            model = LitResNetTransformer.load_from_checkpoint(args.checkpoint)
            model.freeze()
            model = model.to(device)  # Move model to the appropriate device
            transform = ToTensorV2()

            # Import crop function from utils
            from scripts.utils import crop

            # Load and preprocess the image using the same crop function as for training data
            image_path = Path(args.image)
            processed_image = crop(image_path, padding=8)

            if processed_image is None:
                print(f"Error: Could not process image '{args.image}'. It may not contain any text.")
                sys.exit(1)

            # Convert to tensor
            image_tensor = transform(image=np.array(processed_image))["image"]
            image_tensor = image_tensor.unsqueeze(0).float().to(device)  # Move input to the same device

            # Run inference
            with torch.no_grad():  # Use no_grad for inference
                pred = model.model.predict(image_tensor)[0]
                # Move prediction back to CPU for post-processing if needed
                if device.type == "cuda":
                    pred = pred.cpu()
                decoded = model.tokenizer.decode(pred.tolist())
                decoded_str = " ".join(decoded)

            print(f"LaTeX code: {decoded_str}")
        except Exception as e:
            print(f"Error during inference: {e}")
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
