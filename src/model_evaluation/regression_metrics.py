"""Evaluate model on test using regression metrics

Metrics:
- Mean squared error
- Mean absolute error
- Root mean squared error
- Mean absolute percentage error
- R2 score
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,r2_score, root_mean_squared_error, mean_absolute_percentage_error
)
from utils import LoggerMixin

class RegressionMetrics(LoggerMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.logger = self.setup_class_logger("RegressionMetrics", config, "logging")

    def metrics(self, y_true: pd.DataFrame | np.ndarray, y_pred: pd.DataFrame | np.ndarray):
        """Evaluate the performance of the model on unseen(test) data
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            """
        self.logger.info(f"="*50)
        self.logger.info(f"STARTING METRIC EVALUATION")
        self.logger.info(f"="*50)

        eval_metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse" : root_mean_squared_error(y_true, y_pred),
            "mae" : mean_absolute_error(y_true, y_pred),
            "mape" : mean_absolute_percentage_error(y_true, y_pred),
            "r2_score" : r2_score(y_pred, y_true)
        }
        return eval_metrics