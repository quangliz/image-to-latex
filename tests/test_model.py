import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
PROJECT_DIRNAME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRNAME))

import torch
from model.resnet_transformer import ResNetTransformer
from model.lit_resnet_transformer import LitResNetTransformer


def test_resnet_transformer():
    """Test the ResNetTransformer model."""
    # Create a small model for testing
    model = ResNetTransformer(
        d_model=32,
        dim_feedforward=64,
        nhead=2,
        dropout=0.1,
        num_decoder_layers=2,
        max_output_len=10,
        sos_index=0,
        eos_index=1,
        pad_index=2,
        num_classes=10,
    )
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 64, 128)  # (B, C, H, W)
    y = torch.randint(0, 10, (batch_size, 5))  # (B, Sy)
    
    output = model(x, y)
    
    # Check output shape
    assert output.shape == (batch_size, 10, 5), f"Expected shape (2, 10, 5), got {output.shape}"
    
    # Test predict method
    with torch.no_grad():
        predictions = model.predict(x)
    
    # Check predictions shape
    assert predictions.shape[0] == batch_size, f"Expected batch size {batch_size}, got {predictions.shape[0]}"
    assert predictions.shape[1] <= model.max_output_len, f"Expected max length {model.max_output_len}, got {predictions.shape[1]}"
    
    print("ResNetTransformer tests passed!")


def test_lit_resnet_transformer():
    """Test the LitResNetTransformer model."""
    # This test requires the vocab.json file to be present
    vocab_file = PROJECT_DIRNAME / "data" / "vocab.json"
    if not vocab_file.exists():
        print(f"Skipping LitResNetTransformer test: {vocab_file} not found")
        return
    
    # Create a small model for testing
    model = LitResNetTransformer(
        d_model=32,
        dim_feedforward=64,
        nhead=2,
        dropout=0.1,
        num_decoder_layers=2,
        max_output_len=10,
    )
    
    # Test that the model was initialized correctly
    assert model.model is not None, "Model was not initialized"
    assert model.tokenizer is not None, "Tokenizer was not initialized"
    
    print("LitResNetTransformer initialization test passed!")


if __name__ == "__main__":
    test_resnet_transformer()
    test_lit_resnet_transformer()
