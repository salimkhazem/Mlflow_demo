"""
Unit tests for model module.
"""

import pytest
import torch
from src.model import create_model, MNISTModel

def test_simple_model_creation():
    """Test simple model creation."""
    model = create_model("simple", hidden_nodes=64)
    assert isinstance(model, MNISTModel)
    assert model.hidden_nodes == 64

def test_model_forward_pass():
    """Test model forward pass."""
    model = create_model("simple", hidden_nodes=32)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 10)
    assert not torch.isnan(output).any()

def test_model_parameters():
    """Test model has correct number of parameters."""
    model = create_model("simple", hidden_nodes=64)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Expected: (784 * 64) + 64 + (64 * 10) + 10 = 50,890
    expected_params = (784 * 64) + 64 + (64 * 10) + 10
    assert total_params == expected_params

@pytest.mark.parametrize("model_type", ["simple", "conv"])
def test_model_types(model_type):
    """Test different model types."""
    model = create_model(model_type)
    assert model is not None
    
    # Test forward pass
    input_tensor = torch.randn(2, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (2, 10) 