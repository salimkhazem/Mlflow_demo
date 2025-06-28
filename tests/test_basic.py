"""
Basic tests for the MNIST project.
"""

import sys
import os
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_config_functionality():
    """Test configuration functionality."""
    from config import Params, get_config
    
    # Test basic config creation
    config = Params(batch_size=64, epochs=5)
    assert config.batch_size == 64
    assert config.epochs == 5
    
    # Test config dictionary conversion
    config_dict = config.to_dict()
    assert config_dict['batch_size'] == 64
    assert config_dict['epochs'] == 5
    
    # Test environment configs
    dev_config = get_config("dev")
    assert dev_config.batch_size == 128
    assert dev_config.epochs == 2
    
    prod_config = get_config("prod")
    assert prod_config.batch_size == 256
    assert prod_config.epochs == 10
    
    print("✅ Config functionality test passed")


def test_model_functionality():
    """Test model functionality."""
    import torch
    from model import create_model, MNISTModel
    
    # Test simple model creation
    model = create_model("simple", hidden_nodes=32)
    assert isinstance(model, MNISTModel)
    assert model.hidden_nodes == 32
    
    # Test model info
    model_info = model.get_model_info()
    assert model_info['hidden_nodes'] == 32
    assert model_info['num_classes'] == 10
    assert model_info['total_parameters'] > 0
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 28, 28)
    output = model(test_input)
    
    assert output.shape == (batch_size, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    print("✅ Model functionality test passed")


def test_dataset_functionality():
    """Test dataset functionality (without downloading data)."""
    from dataset import get_mnist_transforms, MNISTDataModule
    from config import Params
    import torch
    
    # Test transforms
    transform = get_mnist_transforms()
    assert transform is not None
    
    # Test transform application
    dummy_image = torch.randn(28, 28)  # Single channel image
    transformed = transform(dummy_image)
    assert transformed.shape == (1, 28, 28)
    
    # Test data module creation
    config = Params(batch_size=32, seed=42)
    data_module = MNISTDataModule(config, data_dir='./test_data')
    
    assert data_module.config.batch_size == 32
    assert data_module.config.seed == 42
    
    # Test data info
    info = data_module.get_data_info()
    assert info['num_classes'] == 10
    assert info['input_shape'] == (1, 28, 28)
    assert info['batch_size'] == 32
    
    print("✅ Dataset functionality test passed")


if __name__ == "__main__":
    test_config_functionality()
    test_model_functionality()
    test_dataset_functionality()
    print("🎉 All tests passed!") 