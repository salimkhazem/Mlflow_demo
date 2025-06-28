# 🤖 MNIST MLOps Pipeline

A complete MLOps pipeline for MNIST digit classification with MLflow experiment tracking, automated CI/CD, and production-ready deployment capabilities.

![CI Status](https://github.com/your-username/mlflow_mnist_2/workflows/Continuous%20Integration/badge.svg)
![Model Training](https://github.com/your-username/mlflow_mnist_2/workflows/Model%20Training/badge.svg)

## 🚀 Features

- **🔄 Complete MLOps Pipeline**: End-to-end machine learning workflow
- **📊 MLflow Integration**: Experiment tracking, model registry, and artifact management
- **🔧 CI/CD with GitHub Actions**: Automated testing, training, and deployment
- **📈 Rich Visualizations**: Training metrics, confusion matrices, and sample predictions
- **🏗️ Modular Architecture**: Clean, maintainable, and extensible codebase
- **🐳 Docker Support**: Containerized deployment ready
- **⚙️ Configuration Management**: Environment-specific configurations
- **🧪 Comprehensive Testing**: Unit tests, integration tests, and code quality checks

## 📁 Project Structure 
mlflow_mnist_2/
├── .github/
│ └── workflows/
│ ├── ci.yml # Continuous Integration
│ ├── model-training.yml # Model Training Pipeline
│ └── cd.yml # Continuous Deployment
├── src/
│ ├── init.py
│ ├── config.py # Configuration management
│ ├── dataset.py # Data loading and preprocessing
│ ├── model.py # Neural network models
│ ├── train.py # Training pipeline
│ ├── visualization.py # Plotting and visualization
│ ├── mlflow_setup.py # MLflow server management
│ ├── main.py # Main entry point
│ └── utils.py # Utility functions
├── tests/
│ ├── init.py
│ ├── test_basic.py # Basic functionality tests
│ └── test_.py # Additional test files
├── scripts/
│ ├── validate_model.py # Model validation
│ ├── generate_report.py # Report generation
│ └── deploy.py # Deployment scripts
├── requirements.txt # Python dependencies
├── Dockerfile # Container configuration
├── pytest.ini # Test configuration
└── README.md # This file