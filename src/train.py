"""
Training module for MNIST model.

This module contains training and evaluation functions with MLflow integration
for experiment tracking and model management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader
from pathlib import Path
from visualization import (
    plot_mnist_samples,
    plot_prediction_samples,
    plot_misclassified_samples,
    plot_class_distribution,
    plot_training_metrics
)

# Handle imports for both relative and absolute execution
try:
    from .config import Params
    from .model import create_model
    from .dataset import MNISTDataModule
    from .mlflow_setup import MLflowManager
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import Params
    from model import create_model
    from dataset import MNISTDataModule
    from mlflow_setup import MLflowManager

logger = logging.getLogger(__name__)


class MNISTTrainer:
    """
    Trainer class for MNIST model with MLflow integration.
    
    This class handles the complete training pipeline including data loading,
    model training, evaluation, and experiment tracking with MLflow.
    
    Args:
        config (Params): Configuration parameters.
        model_type (str): Type of model to use ('simple' or 'conv').
        experiment_name (str): MLflow experiment name.
    
    Example:
        >>> config = Params(epochs=5, batch_size=128)
        >>> trainer = MNISTTrainer(config, model_type="simple")
        >>> trainer.train()
    """
    
    def __init__(
        self,
        config: Params,
        model_type: str = "simple",
        experiment_name: str = "MNIST_Experiments"
    ):
        self.config = config
        self.model_type = model_type
        self.experiment_name = experiment_name
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.data_module = None
        
        logger.info(f"Trainer initialized with device: {self.device}")
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup(self) -> None:
        """Setup model, optimizer, and data loaders."""
        # Create model
        model_kwargs = {"hidden_nodes": self.config.hidden_nodes} if self.model_type == "simple" else {}
        self.model = create_model(self.model_type, **model_kwargs)
        self.model.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup data
        self.data_module = MNISTDataModule(self.config, data_dir=self.config.data_dir)
        
        logger.info("Training setup completed")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader.
            epoch (int): Current epoch number.
        
        Returns:
            Dict[str, float]: Training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                current_loss = loss.item()
                current_acc = 100.0 * correct / total
                
                logger.info(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                    f'Loss: {current_loss:.6f}\tAcc: {current_acc:.2f}%'
                )
                
                # Log to MLflow
                step = epoch * len(train_loader) + batch_idx
                mlflow.log_metric('train_loss_step', current_loss, step=step)
                mlflow.log_metric('train_accuracy_step', current_acc, step=step)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }
    
    def evaluate(self, test_loader: DataLoader, epoch: int) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader (DataLoader): Test data loader.
            epoch (int): Current epoch number.
        
        Returns:
            Dict[str, Any]: Evaluation metrics and confusion matrix.
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        confusion_matrix = np.zeros((10, 10), dtype=int)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Sum up batch loss
                test_loss += self.criterion(output, target).item()
                
                # Get predictions
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update confusion matrix
                for t, p in zip(target.cpu().numpy(), pred.cpu().numpy().flatten()):
                    confusion_matrix[t, p] += 1
        
        # Calculate metrics
        avg_loss = test_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        logger.info(
            f'Test set: Average loss: {avg_loss:.4f}, '
            f'Accuracy: {correct}/{total} ({accuracy:.2f}%)'
        )
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'confusion_matrix': confusion_matrix
        }
    
    def create_confusion_matrix_plot(self, confusion_matrix: np.ndarray, epoch: int) -> str:
        """
        Create and save confusion matrix plot.
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix.
            epoch (int): Current epoch number.
        
        Returns:
            str: Path to saved plot.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(10),
            yticklabels=range(10)
        )
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = f"confusion_matrix_epoch_{epoch}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def log_visualizations(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        """
        Create and log various visualizations to MLflow.
        
        Args:
            train_loader (DataLoader): Training data loader.
            test_loader (DataLoader): Test data loader.
        """
        logger.info("Creating and logging visualizations...")
        
        try:
            # 1. Plot training samples
            plot_mnist_samples(
                train_loader, 
                num_samples=16, 
                save_path="train_samples.png",
                log_to_mlflow=True
            )
            
            # 2. Plot test samples
            plot_mnist_samples(
                test_loader, 
                num_samples=16, 
                save_path="test_samples.png",
                log_to_mlflow=True
            )
            
            # 3. Plot class distribution for training data
            plot_class_distribution(
                train_loader,
                save_path="train_class_distribution.png",
                log_to_mlflow=True
            )
            
            # 4. Plot class distribution for test data
            plot_class_distribution(
                test_loader,
                save_path="test_class_distribution.png",
                log_to_mlflow=True
            )
            
            # 5. Plot model predictions (after training)
            if self.model is not None:
                plot_prediction_samples(
                    self.model,
                    test_loader,
                    self.device,
                    num_samples=16,
                    save_path="model_predictions.png",
                    log_to_mlflow=True
                )
                
                # 6. Plot misclassified samples
                plot_misclassified_samples(
                    self.model,
                    test_loader,
                    self.device,
                    num_samples=16,
                    save_path="misclassified_samples.png",
                    log_to_mlflow=True
                )
            
            logger.info("✅ All visualizations logged to MLflow successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {e}")
    
    def train(self) -> Dict[str, Any]:
        """Complete training pipeline with visualizations."""
        # Setup
        self.setup()
        train_loader, test_loader = self.data_module.get_data_loaders()
        
        # MLflow experiment setup
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run():
            # Log parameters
            for key, value in self.config.to_dict().items():
                mlflow.log_param(key, value)
            
            mlflow.log_param('model_type', self.model_type)
            mlflow.log_param('device', str(self.device))
            
            # Log model info if available
            if hasattr(self.model, 'get_model_info'):
                model_info = self.model.get_model_info()
                for key, value in model_info.items():
                    mlflow.log_param(f'model_{key}', value)
            
            # Log initial visualizations (before training)
            logger.info("Logging initial data visualizations...")
            plot_mnist_samples(train_loader, num_samples=16, save_path="train_samples.png")
            plot_mnist_samples(test_loader, num_samples=16, save_path="test_samples.png")
            plot_class_distribution(train_loader, save_path="train_class_distribution.png")
            plot_class_distribution(test_loader, save_path="test_class_distribution.png")
            
            # Training loop
            best_accuracy = 0.0
            training_history = []
            
            for epoch in range(1, self.config.epochs + 1):
                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Evaluate
                eval_metrics = self.evaluate(test_loader, epoch)
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **eval_metrics, 'epoch': epoch}
                training_history.append(epoch_metrics)
                
                # Log metrics to MLflow
                mlflow.log_metric('train_loss', train_metrics['train_loss'], step=epoch)
                mlflow.log_metric('train_accuracy', train_metrics['train_accuracy'], step=epoch)
                mlflow.log_metric('test_loss', eval_metrics['test_loss'], step=epoch)
                mlflow.log_metric('test_accuracy', eval_metrics['test_accuracy'], step=epoch)
                
                # Save best model
                if eval_metrics['test_accuracy'] > best_accuracy:
                    best_accuracy = eval_metrics['test_accuracy']
                    mlflow.log_metric('best_accuracy', best_accuracy)
                
                # Log training progress plot every few epochs
                if epoch % 2 == 0 or epoch == self.config.epochs:
                    plot_training_metrics(
                        training_history,
                        save_path=f"training_progress_epoch_{epoch}.png"
                    )
                
                # Create and log confusion matrix for final epoch
                if epoch == self.config.epochs:
                    plot_path = self.create_confusion_matrix_plot(
                        eval_metrics['confusion_matrix'], epoch
                    )
                    mlflow.log_artifact(plot_path)
                    Path(plot_path).unlink()  # Clean up
            
            # Log post-training visualizations
            logger.info("Logging post-training visualizations...")
            plot_prediction_samples(
                self.model, test_loader, self.device, 
                num_samples=16, save_path="final_predictions.png"
            )
            plot_misclassified_samples(
                self.model, test_loader, self.device,
                num_samples=16, save_path="final_misclassified.png"
            )
            
            # Log final training metrics plot
            plot_training_metrics(training_history, save_path="final_training_metrics.png")
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            # Cleanup temporary files
            temp_files = [
                "train_samples.png", "test_samples.png",
                "train_class_distribution.png", "test_class_distribution.png",
                "final_predictions.png", "final_misclassified.png",
                "final_training_metrics.png"
            ]
            
            for temp_file in temp_files:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            
            # Final results
            results = {
                'best_accuracy': best_accuracy,
                'final_train_loss': training_history[-1]['train_loss'],
                'final_test_loss': training_history[-1]['test_loss'],
                'training_history': training_history
            }
            
            logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
            return results


def hyperparameter_sweep(base_config: Params, param_grid: Dict[str, list]) -> None:
    """
    Perform hyperparameter sweep.
    
    Args:
        base_config (Params): Base configuration.
        param_grid (Dict[str, list]): Parameter grid for sweep.
    
    Example:
        >>> param_grid = {
        ...     'lr': [0.01, 0.02, 0.05],
        ...     'momentum': [0.9, 0.95],
        ...     'hidden_nodes': [32, 64, 128]
        ... }
        >>> hyperparameter_sweep(base_config, param_grid)
    """
    import itertools
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    logger.info(f"Starting hyperparameter sweep with {len(combinations)} combinations")
    
    for i, combination in enumerate(combinations):
        # Create config for this combination
        config_dict = dict(zip(keys, combination))
        config = Params(**{**base_config.to_dict(), **config_dict})
        
        logger.info(f"Running combination {i+1}/{len(combinations)}: {config_dict}")
        
        # Train with this configuration
        trainer = MNISTTrainer(config, experiment_name="MNIST_Hyperparameter_Sweep")
        trainer.train()


if __name__ == "__main__":
    # Example usage
    import warnings
    warnings.filterwarnings("ignore")
    
    # Create configuration
    config = Params(
        batch_size=128,
        epochs=3,
        lr=0.01,
        momentum=0.9,
        hidden_nodes=64
    )
    
    print("Starting MNIST training...")
    
    # Train model
    trainer = MNISTTrainer(config, model_type="simple")
    results = trainer.train()
    
    print(f"Training completed! Best accuracy: {results['best_accuracy']:.2f}%") 