"""Predict the values of y using X_test and the trained model instance"""


import pandas as pd
import numpy as np
from typing import Optional
from utils import LoggerMixin

class RunPredictions(LoggerMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("RunPredictions",config, "logging")
        self.y_pred = None 

    def predictions(self, model, X_test: pd.DataFrame | np.ndarray) -> Optional[pd.Series | np.ndarray]:
        """Run predictions on held_out test set
        
        Args:
            model: trained model instance
            X_test: Held-out test set without target variable
            
        Returns:
            A one-dimensional array or None if X_test is empty
            """

        target_variable = self.config.get("target_column", "selling_price_log")

        if len(X_test) == 0:
            return pd.Series()
        
        if target_variable in X_test.columns:
            raise ValueError(f"{target_variable} must not be present in test_set")
        
        try:
            y_pred = model.predict(X_test)
            return y_pred 
        
        except Exception as e:
            self.logger.error(f"Error encountered during predictions: {e}")

