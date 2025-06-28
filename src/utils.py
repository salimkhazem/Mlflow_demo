"""
Utility functions for the MNIST MLflow pipeline.

This module contains helper functions for data visualization, model analysis,
and experiment management.
"""

import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def visualize_mnist_samples(data_loader, num_samples: int = 8) -> None:
    """
    Visualize random MNIST samples from data loader.
    
    Args:
        data_loader: PyTorch DataLoader with MNIST data.
        num_samples (int): Number of samples to display.
    """
    # Get a batch of data
    data, labels = next(iter(data_loader))
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(data))):
        img = data[i].squeeze().numpy()
        label = labels[i].item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history: List[Dict[str, Any]]) -> None:
    """
    Plot training history metrics.
    
    Args:
        history (List[Dict]): Training history with metrics per epoch.
    """
    df = pd.DataFrame(history)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plots
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss')
    ax1.plot(df['epoch'], df['test_loss'], 'r-', label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plots
    ax2.plot(df['epoch'], df['train_accuracy'], 'b-', label='Train Accuracy')
    ax2.plot(df['epoch'], df['test_accuracy'], 'r-', label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Loss difference
    ax3.plot(df['epoch'], df['train_loss'] - df['test_loss'], 'g-')
    ax3.set_title('Loss Difference (Train - Test)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.grid(True)
    
    # Accuracy difference
    ax4.plot(df['epoch'], df['train_accuracy'] - df['test_accuracy'], 'g-')
    ax4.set_title('Accuracy Difference (Train - Test)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference (%)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_model_weights(model: torch.nn.Module, layer_name: str = "classifier.0") -> None:
    """
    Analyze and visualize model weights.
    
    Args:
        model (torch.nn.Module): Trained model.
        layer_name (str): Name of layer to analyze.
    """
    # Get weights
    weights = None
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            weights = param.data.cpu().numpy()
            break
    
    if weights is None:
        logger.warning(f"Layer {layer_name} not found in model")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Weight distribution
    ax1.hist(weights.flatten(), bins=50, alpha=0.7)
    ax1.set_title('Weight Distribution')
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Frequency')
    
    # Weight heatmap
    sns.heatmap(np.abs(weights), ax=ax2, cmap='viridis')
    ax2.set_title('Absolute Weight Values')
    
    # Weight statistics by neuron
    weight_stats = np.array([
        np.mean(np.abs(weights), axis=1),
        np.std(weights, axis=1),
        np.max(np.abs(weights), axis=1)
    ]).T
    
    ax3.plot(weight_stats[:, 0], label='Mean |Weight|')
    ax3.plot(weight_stats[:, 1], label='Std Weight')
    ax3.plot(weight_stats[:, 2], label='Max |Weight|')
    ax3.set_title('Weight Statistics by Neuron')
    ax3.set_xlabel('Neuron Index')
    ax3.legend()
    
    # Weight correlation
    correlation = np.corrcoef(weights)
    sns.heatmap(correlation, ax=ax4, cmap='coolwarm', center=0)
    ax4.set_title('Weight Correlation Matrix')
    
    plt.tight_layout()
    plt.show()


def compare_experiments(experiment_name: str, metric: str = "test_accuracy") -> pd.DataFrame:
    """
    Compare experiments from MLflow.
    
    Args:
        experiment_name (str): Name of MLflow experiment.
        metric (str): Metric to compare.
    
    Returns:
        pd.DataFrame: Comparison results.
    """
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment '{experiment_name}' not found")
            return pd.DataFrame()
        
        # Get runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            logger.warning(f"No runs found in experiment '{experiment_name}'")
            return pd.DataFrame()
        
        # Select relevant columns
        columns = ['run_id', 'start_time', 'end_time', 'status']
        
        # Add parameter columns
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        columns.extend(param_cols)
        
        # Add metric columns
        metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
        columns.extend(metric_cols)
        
        # Filter and sort
        result_df = runs[columns].copy()
        if f'metrics.{metric}' in result_df.columns:
            result_df = result_df.sort_values(f'metrics.{metric}', ascending=False)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error comparing experiments: {e}")
        return pd.DataFrame()


def export_best_model(experiment_name: str, output_dir: str = "best_models") -> Optional[str]:
    """
    Export the best model from an experiment.
    
    Args:
        experiment_name (str): Name of MLflow experiment.
        output_dir (str): Directory to save the model.
    
    Returns:
        Optional[str]: Path to exported model, None if failed.
    """
    try:
        # Compare experiments to find best
        comparison = compare_experiments(experiment_name, "test_accuracy")
        
        if comparison.empty:
            logger.error("No experiments found to export")
            return None
        
        # Get best run
        best_run_id = comparison.iloc[0]['run_id']
        best_accuracy = comparison.iloc[0]['metrics.test_accuracy']
        
        logger.info(f"Best run: {best_run_id} with accuracy: {best_accuracy:.2f}%")
        
        # Download model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = mlflow.artifacts.download_artifacts(
            run_id=best_run_id,
            artifact_path="model",
            dst_path=str(output_path)
        )
        
        logger.info(f"Model exported to: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        return None


def create_experiment_report(experiment_name: str, output_file: str = "experiment_report.html") -> None:
    """
    Create an HTML report of experiment results.
    
    Args:
        experiment_name (str): Name of MLflow experiment.
        output_file (str): Output HTML file path.
    """
    try:
        # Get experiment data
        comparison = compare_experiments(experiment_name)
        
        if comparison.empty:
            logger.error("No data found for report")
            return
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MLflow Experiment Report: {experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <h1>MLflow Experiment Report</h1>
            <h2>Experiment: {experiment_name}</h2>
            <h3>Summary</h3>
            <ul>
                <li>Total Runs: {len(comparison)}</li>
                <li>Best Accuracy: {comparison.iloc[0]['metrics.test_accuracy']:.2f}%</li>
                <li>Date Range: {comparison['start_time'].min()} to {comparison['start_time'].max()}</li>
            </ul>
            
            <h3>Results Table</h3>
            {comparison.to_html(classes='table', table_id='results')}
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
    except Exception as e:
        logger.error(f"Error creating experiment report: {e}") 