"""Evaluate model on test using regression metrics

Metrics:
- Mean squared error
- Mean absolute error
- Root mean squared error
- Mean absolute percentage error
- R2 score
"""
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,r2_score, root_mean_squared_error, mean_absolute_percentage_error
)
from utils import LoggerMixin

class RegressionMetrics(LoggerMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.logger = self.setup_class_logger("RegressionMetrics", config, "logging")
        self.metrics_ = self.config.get("regression_metrics", {})

    def metrics(self):
        """Evaluate the performance of the model on unseen(test) data"""
        for metric in self.metrics_:
            if 
    