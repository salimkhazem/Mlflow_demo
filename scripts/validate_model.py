"""
Model validation script for CI/CD pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import mlflow.pytorch
from src.config import get_config
from src.dataset import MNISTDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_model(model_path: str, accuracy_threshold: float = 90.0) -> bool:
    """
    Validate model performance against threshold.
    
    Args:
        model_path (str): Path to model directory.
        accuracy_threshold (float): Minimum accuracy threshold.
    
    Returns:
        bool: True if model passes validation.
    """
    try:
        # Load model
        model = mlflow.pytorch.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Setup data
        config = get_config("dev")
        data_module = MNISTDataModule(config)
        _, test_loader = data_module.get_data_loaders()
        
        # Evaluate model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f"Model accuracy: {accuracy:.2f}%")
        
        if accuracy >= accuracy_threshold:
            logger.info(f"✅ Model validation passed (accuracy >= {accuracy_threshold}%)")
            return True
        else:
            logger.error(f"❌ Model validation failed (accuracy < {accuracy_threshold}%)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model validation error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate trained model")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--threshold", type=float, default=90.0, help="Accuracy threshold")
    
    args = parser.parse_args()
    
    success = validate_model(args.model_path, args.threshold)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 