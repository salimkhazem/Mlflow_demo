"""
Main pipeline script for MNIST MLflow experiment.

This script orchestrates the complete MLOps pipeline including MLflow setup,
data loading, model training, and experiment tracking.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, Any

# Handle imports
try:
    from .config import get_config, Params
    from .train import MNISTTrainer, hyperparameter_sweep
    from .mlflow_setup import MLflowManager
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import get_config, Params
    from train import MNISTTrainer, hyperparameter_sweep
    from mlflow_setup import MLflowManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


def run_single_experiment(config: Params, model_type: str = "simple") -> Dict[str, Any]:
    """
    Run a single training experiment.
    
    Args:
        config (Params): Configuration parameters.
        model_type (str): Type of model to use.
    
    Returns:
        Dict[str, Any]: Training results.
    """
    logger.info("Starting single experiment...")
    logger.info(f"Configuration: {config.to_dict()}")
    
    trainer = MNISTTrainer(config, model_type=model_type)
    results = trainer.train()
    
    logger.info(f"Experiment completed. Best accuracy: {results['best_accuracy']:.2f}%")
    return results


def run_hyperparameter_sweep(base_config: Params) -> None:
    """
    Run hyperparameter sweep experiment.
    
    Args:
        base_config (Params): Base configuration.
    """
    logger.info("Starting hyperparameter sweep...")
    
    # Define parameter grid
    param_grid = {
        'lr': [0.01, 0.02, 0.05],
        'momentum': [0.9, 0.95, 0.99],
        'hidden_nodes': [32, 48, 64]
    }
    
    hyperparameter_sweep(base_config, param_grid)
    logger.info("Hyperparameter sweep completed!")


def main():
    """Main function to run the MLOps pipeline."""
    parser = argparse.ArgumentParser(description="MNIST MLflow Pipeline")
    parser.add_argument(
        "--env", 
        type=str, 
        default="default",
        choices=["default", "dev", "prod"],
        help="Environment configuration to use"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="simple",
        choices=["simple", "conv"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "sweep"],
        help="Run mode: single experiment or hyperparameter sweep"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow UI setup"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.env)
    
    # Override config with command line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    
    logger.info(f"Using {args.env} configuration")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Mode: {args.mode}")
    
    # Setup MLflow if requested
    mlflow_manager = None
    if not args.no_mlflow:
        try:
            logger.info("Setting up MLflow with ngrok...")
            mlflow_manager = MLflowManager(config.ngrok_token)
            url = mlflow_manager.start_complete_setup()
            logger.info(f"✅ MLflow UI available at: {url}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            logger.info("Continuing without MLflow UI...")
    
    try:
        # Run experiment based on mode
        if args.mode == "single":
            results = run_single_experiment(config, args.model_type)
            logger.info("🎉 Single experiment completed successfully!")
            
        elif args.mode == "sweep":
            run_hyperparameter_sweep(config)
            logger.info("🎉 Hyperparameter sweep completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        raise
    
    finally:
        # Cleanup MLflow
        if mlflow_manager:
            logger.info("Shutting down MLflow...")
            mlflow_manager.stop()


if __name__ == "__main__":
    main() 