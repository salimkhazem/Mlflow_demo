"""
Neural network models for MNIST classification.

This module contains model definitions, model factory functions, and utilities
for creating and managing PyTorch models for MNIST digit classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MNISTModel(nn.Module):
    """
    Simple feedforward neural network for MNIST classification.
    
    This model consists of a simple classifier with configurable hidden layers
    for digit classification on the MNIST dataset.
    
    Args:
        hidden_nodes (int): Number of nodes in the hidden layer.
        input_size (int): Size of input features (default: 784 for 28x28 images).
        num_classes (int): Number of output classes (default: 10 for digits).
        dropout_rate (float): Dropout rate for regularization.
    
    Example:
        >>> model = MNISTModel(hidden_nodes=64)
        >>> output = model(torch.randn(32, 1, 28, 28))
        >>> print(output.shape)  # torch.Size([32, 10])
    """
    
    def __init__(
        self, 
        hidden_nodes: int = 32,
        input_size: int = 784,
        num_classes: int = 10,
        dropout_rate: float = 0.2
    ):
        super(MNISTModel, self).__init__()
        
        self.hidden_nodes = hidden_nodes
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_nodes, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Flatten the input
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'hidden_nodes': self.hidden_nodes,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class ConvMNISTModel(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.
    
    A more sophisticated CNN model for better performance on MNIST.
    
    Args:
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super(ConvMNISTModel, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout and pooling
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN."""
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def create_model(model_type: str = "simple", **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type (str): Type of model to create ('simple' or 'conv').
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Initialized model.
    
    Raises:
        ValueError: If model_type is not supported.
    
    Example:
        >>> model = create_model("simple", hidden_nodes=64)
        >>> model = create_model("conv", dropout_rate=0.3)
    """
    models = {
        "simple": MNISTModel,
        "conv": ConvMNISTModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    model = models[model_type](**kwargs)
    logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def save_model(model: nn.Module, filepath: str, metadata: Optional[Dict] = None) -> None:
    """
    Save model with metadata.
    
    Args:
        model (nn.Module): Model to save.
        filepath (str): Path to save the model.
        metadata (Optional[Dict]): Additional metadata to save.
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'metadata': metadata or {}
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str, model_class: nn.Module, **model_kwargs) -> nn.Module:
    """
    Load model from file.
    
    Args:
        filepath (str): Path to model file.
        model_class (nn.Module): Model class to instantiate.
        **model_kwargs: Arguments for model initialization.
    
    Returns:
        nn.Module: Loaded model.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model loaded from {filepath}")
    return model


class Model(nn.Module):
    def __init__(self, nH = 32): 
        super(Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(784, nH),  # 28 x 28 = 784
            nn.ReLU(),
            nn.Linear(nH, 10)
        )
             
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
# Backward compatibility - alias for the original Model class
Model = MNISTModel


if __name__ == "__main__":
    # Test models
    print("Testing MNIST models...")
    
    # Test simple model
    simple_model = create_model("simple", hidden_nodes=64)
    print(f"Simple model info: {simple_model.get_model_info()}")
    
    # Test conv model
    conv_model = create_model("conv")
    
    # Test forward pass
    test_input = torch.randn(4, 1, 28, 28)
    
    simple_output = simple_model(test_input)
    conv_output = conv_model(test_input)
    
    print(f"Simple model output shape: {simple_output.shape}")
    print(f"Conv model output shape: {conv_output.shape}")
    
    print("✅ All model tests passed!")