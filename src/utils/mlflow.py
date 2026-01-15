"""  
centralized mlflow utility functions.
Provides functions to set tracking URIs, experiments, and manage runs
"""

import logging
from typing import Optional 
import mlflow
from mlflow import MlflowClient
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)

def set_tracking_uri(uri: str) -> None:
    """
    Set the MLflow tracking URI.
    
    Args:
        uri: Tracking URI (e.g., 'http://localhost:5000' or 'file:///path/to/mlruns')
    """
    try:
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow tracking URI set to: {uri}")
    except Exception as e:
        logger.error(f"Failed to set MLflow tracking URI: {e}")
        raise


def set_experiment(experiment_name: str) -> None:
    """
    Set the MLflow experiment.

    Args:
        experiment_name: Name of the experiment to set
    """
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise
