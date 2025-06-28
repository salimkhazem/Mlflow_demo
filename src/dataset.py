"""
Dataset module for MNIST data loading and preprocessing.

This module provides functions and classes for loading, preprocessing, and creating
data loaders for the MNIST dataset with proper reproducibility controls.
"""

import torch
import logging
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .config import Params
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import Params

# Set up logging
logger = logging.getLogger(__name__)


def get_mnist_transforms(normalize_mean: float = 0.1307, normalize_std: float = 0.3081) -> transforms.Compose:
    """
    Create standardized transforms for MNIST dataset.
    
    Args:
        normalize_mean (float): Mean value for normalization. Default is MNIST standard.
        normalize_std (float): Standard deviation for normalization. Default is MNIST standard.
    
    Returns:
        transforms.Compose: Composed transforms for MNIST preprocessing.
    
    Example:
        >>> transform = get_mnist_transforms()
        >>> # Apply to dataset
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((normalize_mean,), (normalize_std,))
    ])


def load_mnist_datasets(
    data_dir: str = '../data',
    transform: Optional[transforms.Compose] = None,
    download: bool = True
) -> Tuple[datasets.MNIST, datasets.MNIST]:
    """
    Load MNIST training and test datasets.
    
    Args:
        data_dir (str): Directory to store/load MNIST data.
        transform (Optional[transforms.Compose]): Transform to apply to data.
            If None, uses default MNIST transforms.
        download (bool): Whether to download data if not present.
    
    Returns:
        Tuple[datasets.MNIST, datasets.MNIST]: Training and test datasets.
    
    Raises:
        RuntimeError: If data loading fails.
    
    Example:
        >>> train_set, test_set = load_mnist_datasets()
        >>> print(f"Training samples: {len(train_set)}")
    """
    if transform is None:
        transform = get_mnist_transforms()
    
    # Ensure data directory exists
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Loading MNIST datasets from {data_dir}")
        
        train_set = datasets.MNIST(
            root=data_dir,
            train=True,
            download=download,
            transform=transform
        )
        
        test_set = datasets.MNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=transform
        )
        
        logger.info(f"Successfully loaded {len(train_set)} training and {len(test_set)} test samples")
        return train_set, test_set
        
    except Exception as e:
        logger.error(f"Failed to load MNIST datasets: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}") from e


def create_data_loaders(
    config: Params,
    data_dir: str = '../data',
    transform: Optional[transforms.Compose] = None,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create reproducible data loaders for training and testing.
    
    Args:
        config (Params): Configuration object containing batch_size and seed.
        data_dir (str): Directory containing MNIST data.
        transform (Optional[transforms.Compose]): Transform to apply to data.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
    
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders.
    
    Example:
        >>> from src.config import Params
        >>> config = Params(batch_size=128, seed=42)
        >>> train_loader, test_loader = create_data_loaders(config)
    """
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load datasets
    train_set, test_set = load_mnist_datasets(
        data_dir=data_dir,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(config.seed)  # Reproducible shuffling
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=False,  # No shuffling for test set
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )
    
    logger.info(f"Created data loaders with batch_size={config.batch_size}, seed={config.seed}")
    return train_loader, test_loader


class MNISTDataModule:
    """
    Data module class for MNIST dataset management.
    
    This class encapsulates all data-related operations for the MNIST dataset,
    providing a clean interface for data loading and preprocessing.
    
    Args:
        config (Params): Configuration object.
        data_dir (str): Directory for MNIST data.
        num_workers (int): Number of data loading workers.
    
    Example:
        >>> from src.config import Params
        >>> config = Params()
        >>> data_module = MNISTDataModule(config)
        >>> train_loader, test_loader = data_module.get_data_loaders()
    """
    
    def __init__(
        self,
        config: Params,
        data_dir: str = '../data',
        num_workers: int = 0
    ):
        self.config = config
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.transform = get_mnist_transforms()
        
        # Initialize datasets as None - lazy loading
        self._train_set: Optional[datasets.MNIST] = None
        self._test_set: Optional[datasets.MNIST] = None
        self._train_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
    
    def setup(self) -> None:
        """Setup datasets and data loaders."""
        if self._train_set is None or self._test_set is None:
            self._train_set, self._test_set = load_mnist_datasets(
                data_dir=self.data_dir,
                transform=self.transform
            )
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get training and test data loaders.
        
        Returns:
            Tuple[DataLoader, DataLoader]: Training and test data loaders.
        """
        if self._train_loader is None or self._test_loader is None:
            self._train_loader, self._test_loader = create_data_loaders(
                config=self.config,
                data_dir=self.data_dir,
                transform=self.transform,
                num_workers=self.num_workers
            )
        
        return self._train_loader, self._test_loader
    
    def get_datasets(self) -> Tuple[datasets.MNIST, datasets.MNIST]:
        """
        Get raw datasets.
        
        Returns:
            Tuple[datasets.MNIST, datasets.MNIST]: Training and test datasets.
        """
        self.setup()
        return self._train_set, self._test_set
    
    def get_data_info(self) -> dict:
        """
        Get information about the datasets.
        
        Returns:
            dict: Dictionary containing dataset information.
        """
        self.setup()
        return {
            'train_size': len(self._train_set),
            'test_size': len(self._test_set),
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'batch_size': self.config.batch_size,
            'transform': str(self.transform)
        }


# Convenience function for quick usage
def get_mnist_loaders(config: Params, data_dir: str = '../data') -> Tuple[DataLoader, DataLoader]:
    """
    Quick function to get MNIST data loaders.
    
    Args:
        config (Params): Configuration object.
        data_dir (str): Data directory path.
    
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders.
    
    Example:
        >>> from src.config import default_config
        >>> train_loader, test_loader = get_mnist_loaders(default_config)
    """
    return create_data_loaders(config, data_dir)


if __name__ == "__main__":
    # Example usage and testing
    try:
        from .config import default_config
    except ImportError:
        from config import default_config
    
    # Test the data loading
    print("Testing MNIST data loading...")
    
    # Create data module
    data_module = MNISTDataModule(default_config)
    
    # Get data loaders
    train_loader, test_loader = data_module.get_data_loaders()
    
    # Print dataset info
    info = data_module.get_data_info()
    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test a batch
    batch_data, batch_labels = next(iter(train_loader))
    print(f"\nBatch shape: {batch_data.shape}")
    print(f"Labels shape: {batch_labels.shape}")
    print(f"Data range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
