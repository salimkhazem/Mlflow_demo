"""
Configuration module for MNIST MLflow experiment.

This module contains configuration classes and parameters for the MNIST model training.
"""

import torch
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Params:
    """Configuration class for hyperparameters and training settings."""
    
    def __init__(
        self, 
        batch_size: int = 256, 
        epochs: int = 4, 
        seed: int = 0, 
        log_interval: int = 200,
        lr: float = 0.01,
        momentum: float = 0.9,
        hidden_nodes: int = 48,
        data_dir: str = '../data',
        ngrok_token: str = "2fZIgz8CYRrl8xQrwNTzEV9Imwx_2vdqT6uyh8rWh8HWDH6w3"
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.log_interval = log_interval
        self.lr = lr
        self.momentum = momentum
        self.hidden_nodes = hidden_nodes
        self.data_dir = data_dir
        self.ngrok_token = ngrok_token
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for MLflow logging."""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'seed': self.seed,
            'log_interval': self.log_interval,
            'lr': self.lr,
            'momentum': self.momentum,
            'hidden_nodes': self.hidden_nodes
        }

# Default configuration
default_config = Params()

# Environment-specific configurations
class DevConfig(Params):
    """Development configuration."""
    def __init__(self):
        super().__init__(
            batch_size=128,
            epochs=2,
            lr=0.01,
            momentum=0.9,
            hidden_nodes=32
        )

class ProdConfig(Params):
    """Production configuration."""
    def __init__(self):
        super().__init__(
            batch_size=256,
            epochs=10,
            lr=0.02,
            momentum=0.95,
            hidden_nodes=48
        )

def get_config(env: str = "default") -> Params:
    """Get configuration based on environment."""
    configs = {
        "default": Params(),
        "dev": DevConfig(),
        "prod": ProdConfig()
    }
    
    if env not in configs:
        raise ValueError(f"Unknown environment: {env}. Available: {list(configs.keys())}")
    
    return configs[env]


