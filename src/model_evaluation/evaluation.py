"""Evaluate model on test using regression metrics

Metrics:
- Mean squared error
- Mean absolute error
- Root mean squared error
- Mean absolute percentage error
- R2 score

Permutation importance
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    root_mean_squared_error, mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from utils import LoggerMixin
import mlflow

class ModelEvaluation(LoggerMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.logger = self.setup_class_logger("ModelEvaluation", config, "logging")

    def metrics(self, y_true: pd.DataFrame | np.ndarray, y_pred: pd.DataFrame | np.ndarray):
        """Evaluate the performance of the model on unseen(test) data
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        """

        eval_metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred) 
        }
        
        # Log metrics
        self.logger.info("\nEvaluation Metrics:")
        for metric_name, metric_value in eval_metrics.items():
            self.logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
        
        return eval_metrics
    
    def permutation_importance(self, model, x_test, y_test, n_repeats=10):
        """Calculate Permutation importance
        
        Args:
            model: Fitted model (can be MLflow wrapped or sklearn)
            x_test: Test features
            y_test: True labels
            n_repeats: Number of shuffles
        """
        if model is None:
            self.logger.error(f"Model cannot be NoneType. Fit model...")
            raise ValueError("Model is None")

        if len(x_test) == 0 or len(y_test) == 0:
            self.logger.error(f"Test set cannot be empty")
            raise ValueError("Test set is empty")

        self.logger.info("\nCalculating permutation importance...")
        
        # Unwrap MLflow model if needed
        if hasattr(model, '_model_impl'):
            self.logger.info("Detected MLflow wrapper, extracting sklearn model...")
            # For sklearn models wrapped by MLflow
            actual_model = model._model_impl.sklearn_model  # Changed from python_model to sklearn_model
        else:
            actual_model = model
        
        perm_importance = permutation_importance(
            actual_model,
            x_test, y_test, 
            n_repeats=n_repeats, 
            random_state=42, 
            n_jobs=-1
        )
        
        # Organize and visualize permutation importance
        sorted_idx = perm_importance.importances_mean.argsort()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(
            perm_importance.importances[sorted_idx].T, 
            vert=False, 
            labels=x_test.columns[sorted_idx]
        )
        ax.set_title("Permutation Importance")
        ax.set_xlabel("Decrease in Model Performance")
        plt.tight_layout()
        
        # Log to MLflow
        mlflow.log_figure(fig, "permutation_importance.png")
        
        plt.close()
        
        self.logger.info("✓ Permutation importance calculated and logged")
        
        mlflow.log_params({
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'importances': perm_importance.importances,
            'sorted_idx': sorted_idx
        })