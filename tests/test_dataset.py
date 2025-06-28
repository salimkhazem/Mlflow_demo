"""
Unit tests for dataset module.
"""

import pytest
import torch
from src.dataset import MNISTDataModule, get_mnist_transforms
from src.config import Params

def test_mnist_transforms():
    """Test MNIST transforms."""
    transform = get_mnist_transforms()
    assert transform is not None

def test_data_module_creation():
    """Test data module creation."""
    config = Params(batch_size=32, seed=42)
    data_module = MNISTDataModule(config)
    assert data_module.config.batch_size == 32
    assert data_module.config.seed == 42

def test_data_loaders():
    """Test data loader creation."""
    config = Params(batch_size=16, seed=42)
    data_module = MNISTDataModule(config)
    
    train_loader, test_loader = data_module.get_data_loaders()
    
    # Test batch from train loader
    train_batch = next(iter(train_loader))
    assert len(train_batch) == 2  # data, labels
    assert train_batch[0].shape[0] == 16  # batch size
    assert train_batch[0].shape[1:] == (1, 28, 28)  # image shape
    
    # Test batch from test loader
    test_batch = next(iter(test_loader))
    assert len(test_batch) == 2
    assert test_batch[0].shape[0] == 16

def test_data_info():
    """Test data info retrieval."""
    config = Params(batch_size=64)
    data_module = MNISTDataModule(config)
    
    info = data_module.get_data_info()
    
    assert info['num_classes'] == 10
    assert info['input_shape'] == (1, 28, 28)
    assert info['batch_size'] == 64 