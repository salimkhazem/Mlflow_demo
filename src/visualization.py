"""
Visualization module for MNIST samples and model analysis with MLflow integration.

This module provides functions to create and log various visualizations
to MLflow for better experiment tracking and analysis.
"""

import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List
from torch.utils.data import DataLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_mnist_samples(
    data_loader: DataLoader, 
    num_samples: int = 16,
    save_path: Optional[str] = None,
    log_to_mlflow: bool = True
) -> str:
    """
    Plot random MNIST samples and optionally log to MLflow.
    
    Args:
        data_loader (DataLoader): DataLoader containing MNIST data.
        num_samples (int): Number of samples to display.
        save_path (Optional[str]): Path to save the plot. If None, auto-generated.
        log_to_mlflow (bool): Whether to log the plot to MLflow.
    
    Returns:
        str: Path to the saved plot.
    
    Example:
        >>> plot_path = plot_mnist_samples(train_loader, num_samples=16)
    """
    # Get a batch of data
    data, labels = next(iter(data_loader))
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('MNIST Sample Images', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot samples
    for i in range(grid_size * grid_size):
        if i < min(num_samples, len(data)):
            img = data[i].squeeze().numpy()
            label = labels[i].item()
            
            axes[i].imshow(img, cmap='gray', interpolation='nearest')
            axes[i].set_title(f'Label: {label}', fontsize=10)
            axes[i].axis('off')
        else:
            # Hide empty subplots
            axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = "mnist_samples.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Log to MLflow
    if log_to_mlflow:
        try:
            mlflow.log_artifact(save_path, "visualizations")
            logger.info(f"Logged sample plot to MLflow: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    plt.close()
    return save_path


def plot_prediction_samples(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 16,
    save_path: Optional[str] = None,
    log_to_mlflow: bool = True
) -> str:
    """
    Plot samples with model predictions and confidence scores.
    
    Args:
        model (torch.nn.Module): Trained model.
        data_loader (DataLoader): DataLoader containing test data.
        device (torch.device): Device to run inference on.
        num_samples (int): Number of samples to display.
        save_path (Optional[str]): Path to save the plot.
        log_to_mlflow (bool): Whether to log the plot to MLflow.
    
    Returns:
        str: Path to the saved plot.
    """
    model.eval()
    
    # Get a batch of data
    data, true_labels = next(iter(data_loader))
    data = data.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(data)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(outputs, dim=1)
        confidence_scores = torch.max(probabilities, dim=1)[0]
    
    # Move to CPU for plotting
    data = data.cpu()
    predicted_labels = predicted_labels.cpu()
    confidence_scores = confidence_scores.cpu()
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle('Model Predictions on Test Samples', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot samples with predictions
    for i in range(grid_size * grid_size):
        if i < min(num_samples, len(data)):
            img = data[i].squeeze().numpy()
            true_label = true_labels[i].item()
            pred_label = predicted_labels[i].item()
            confidence = confidence_scores[i].item()
            
            # Determine color based on correctness
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].imshow(img, cmap='gray', interpolation='nearest')
            axes[i].set_title(
                f'True: {true_label}, Pred: {pred_label}\n'
                f'Confidence: {confidence:.3f}',
                fontsize=9,
                color=color,
                fontweight='bold'
            )
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = "prediction_samples.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Log to MLflow
    if log_to_mlflow:
        try:
            mlflow.log_artifact(save_path, "visualizations")
            logger.info(f"Logged prediction plot to MLflow: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    plt.close()
    return save_path


def plot_misclassified_samples(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 16,
    save_path: Optional[str] = None,
    log_to_mlflow: bool = True
) -> str:
    """
    Plot misclassified samples to analyze model errors.
    
    Args:
        model (torch.nn.Module): Trained model.
        data_loader (DataLoader): DataLoader containing test data.
        device (torch.device): Device to run inference on.
        num_samples (int): Number of misclassified samples to display.
        save_path (Optional[str]): Path to save the plot.
        log_to_mlflow (bool): Whether to log the plot to MLflow.
    
    Returns:
        str: Path to the saved plot.
    """
    model.eval()
    
    misclassified_data = []
    misclassified_true = []
    misclassified_pred = []
    misclassified_conf = []
    
    # Collect misclassified samples
    with torch.no_grad():
        for data, labels in data_loader:
            data_batch = data.to(device)
            outputs = model(data_batch)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            # Find misclassified samples
            mask = predicted.cpu() != labels
            if mask.any():
                misclassified_data.extend(data[mask])
                misclassified_true.extend(labels[mask])
                misclassified_pred.extend(predicted.cpu()[mask])
                misclassified_conf.extend(confidence.cpu()[mask])
                
                # Stop when we have enough samples
                if len(misclassified_data) >= num_samples:
                    break
    
    if not misclassified_data:
        logger.warning("No misclassified samples found!")
        return ""
    
    # Limit to requested number
    num_to_plot = min(num_samples, len(misclassified_data))
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_to_plot)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle('Misclassified Samples', fontsize=16, fontweight='bold', color='red')
    
    # Flatten axes for easier indexing
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot misclassified samples
    for i in range(grid_size * grid_size):
        if i < num_to_plot:
            img = misclassified_data[i].squeeze().numpy()
            true_label = misclassified_true[i].item()
            pred_label = misclassified_pred[i].item()
            confidence = misclassified_conf[i].item()
            
            axes[i].imshow(img, cmap='gray', interpolation='nearest')
            axes[i].set_title(
                f'True: {true_label} → Pred: {pred_label}\n'
                f'Confidence: {confidence:.3f}',
                fontsize=9,
                color='red',
                fontweight='bold'
            )
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = "misclassified_samples.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Log to MLflow
    if log_to_mlflow:
        try:
            mlflow.log_artifact(save_path, "visualizations")
            logger.info(f"Logged misclassified samples to MLflow: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    plt.close()
    return save_path


def plot_class_distribution(
    data_loader: DataLoader,
    save_path: Optional[str] = None,
    log_to_mlflow: bool = True
) -> str:
    """
    Plot class distribution of the dataset.
    
    Args:
        data_loader (DataLoader): DataLoader containing data.
        save_path (Optional[str]): Path to save the plot.
        log_to_mlflow (bool): Whether to log the plot to MLflow.
    
    Returns:
        str: Path to the saved plot.
    """
    # Collect all labels
    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.numpy())
    
    # Count occurrences
    unique, counts = np.unique(all_labels, return_counts=True)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique, counts, color='skyblue', alpha=0.7, edgecolor='navy')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(unique)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics
    total_samples = len(all_labels)
    plt.text(0.02, 0.98, f'Total Samples: {total_samples}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = "class_distribution.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Log to MLflow
    if log_to_mlflow:
        try:
            mlflow.log_artifact(save_path, "visualizations")
            logger.info(f"Logged class distribution to MLflow: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    plt.close()
    return save_path


def plot_training_metrics(
    training_history: List[dict],
    save_path: Optional[str] = None,
    log_to_mlflow: bool = True
) -> str:
    """
    Plot comprehensive training metrics.
    
    Args:
        training_history (List[dict]): Training history with metrics per epoch.
        save_path (Optional[str]): Path to save the plot.
        log_to_mlflow (bool): Whether to log the plot to MLflow.
    
    Returns:
        str: Path to the saved plot.
    """
    epochs = [h['epoch'] for h in training_history]
    train_loss = [h['train_loss'] for h in training_history]
    test_loss = [h['test_loss'] for h in training_history]
    train_acc = [h['train_accuracy'] for h in training_history]
    test_acc = [h['test_accuracy'] for h in training_history]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Dashboard', fontsize=16, fontweight='bold')
    
    # Loss plot
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_loss, 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Loss Over Time', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Accuracy Over Time', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overfitting analysis
    loss_gap = np.array(train_loss) - np.array(test_loss)
    ax3.plot(epochs, loss_gap, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_title('Overfitting Analysis (Train Loss - Test Loss)', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.grid(True, alpha=0.3)
    
    # Learning rate analysis (if available)
    ax4.plot(epochs, test_acc, 'purple', linewidth=3, label='Test Accuracy')
    ax4.set_title('Test Accuracy Trend', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.grid(True, alpha=0.3)
    
    # Add best accuracy annotation
    best_acc_idx = np.argmax(test_acc)
    best_acc = test_acc[best_acc_idx]
    best_epoch = epochs[best_acc_idx]
    ax4.annotate(f'Best: {best_acc:.2f}% (Epoch {best_epoch})',
                xy=(best_epoch, best_acc),
                xytext=(best_epoch + 0.5, best_acc - 2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = "training_metrics.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Log to MLflow
    if log_to_mlflow:
        try:
            mlflow.log_artifact(save_path, "visualizations")
            logger.info(f"Logged training metrics to MLflow: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    plt.close()
    return save_path


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")
    print("✅ Visualization module loaded successfully!") 