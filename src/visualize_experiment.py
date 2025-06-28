"""
Standalone script to create visualizations for existing MLflow experiments.
"""

import mlflow
import argparse
from pathlib import Path
from visualization import *
from dataset import MNISTDataModule
from config import get_config

def visualize_experiment(run_id: str, config_env: str = "default"):
    """
    Create visualizations for an existing MLflow experiment.
    
    Args:
        run_id (str): MLflow run ID to visualize.
        config_env (str): Configuration environment.
    """
    # Load configuration
    config = get_config(config_env)
    
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Setup data
    data_module = MNISTDataModule(config)
    train_loader, test_loader = data_module.get_data_loaders()
    
    # Create visualizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Creating visualizations...")
    
    # Create all visualizations
    plot_mnist_samples(train_loader, save_path=f"viz_train_samples_{run_id}.png", log_to_mlflow=False)
    plot_prediction_samples(model, test_loader, device, save_path=f"viz_predictions_{run_id}.png", log_to_mlflow=False)
    plot_misclassified_samples(model, test_loader, device, save_path=f"viz_misclassified_{run_id}.png", log_to_mlflow=False)
    
    print(f"✅ Visualizations created for run {run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create visualizations for MLflow experiment")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--config", default="default", help="Configuration environment")
    
    args = parser.parse_args()
    visualize_experiment(args.run_id, args.config) 