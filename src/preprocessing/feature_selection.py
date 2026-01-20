"""Feature selection utilities for preprocessing.
Workflow:
1. Import necessary libraries.
2. Use Recursive Feature Elimination (RFE) for feature selection.
3. visualize feature importance.
4. Define fit and transform functions.
5. Save fitted selector as a joblib file
6. Log artifacts using mlflow."""

import pandas as pd
from sklearn.feature_selection import RFECV
from typing import List, Dict, Any
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from utils import LoggerMixin
import joblib

class FeatureSelector(LoggerMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.selector = RFECV(
            estimator=RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), 
            step=1, cv=KFold(5), scoring='neg_mean_squared_error')
        self.logger = self.setup_class_logger("FeatureSelector", config, "logging")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the RFECV selector on the features and target.
        
        Args:
            X (pd.DataFrame): The input features dataframe.
            y (pd.Series): The target variable series.
        """
        self.selector.fit(X, y)
        self.logger.info("Feature selection completed.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input features dataframe using the fitted selector.
        
        Args:
            X (pd.DataFrame): The input features dataframe.
            
        Returns:
            pd.DataFrame: The transformed features dataframe.
        """
        return self.selector.transform(X)

    def save_selector(self, path: str) -> None:
        """Save the fitted selector to a joblib file.
        
        Args:
            path (str): The path to save the selector file.
        """
        joblib.dump(self.selector, path)
        self.logger.info(f"Feature selector saved at {path}.")

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit the selector and transform the input features dataframe.
        
        Args:
            X (pd.DataFrame): The input features dataframe.
            y (pd.Series): The target variable series.
        """
        self.fit(X, y)
        return self.transform(X)
    
    