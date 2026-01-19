"""Feature scaling utilities for preprocessing.

Workflow:
1. Import necessary libraries.
2. Use scikit-learn "standardScaler" for standard scaling.
3. Define fit and transform functions.
4. Save fitted scaler as a joblib file
5. Log artifacts using mlflow.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any 
import numpy as np
from utils import LoggerMixin
import joblib
import mlflow

class FeatureScaler(LoggerMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.scaler = StandardScaler | None = None
        self.logger = self.setup_class_logger("FeatureScaler", config, "logging")

    def fit(self, df: pd.DataFrame, numerical_columns: List[str]) -> None:
        """Fit the StandardScaler on the numerical columns.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            numerical_columns (List[str]): List of numerical column names to scale.
            
        Returns:
            None
        """
        self.scaler = StandardScaler()
        self.scaler.fit(df[numerical_columns])
        self.save_scaler(self.config['scaling']['artifact_dir'] + "artifacts/standard_scaler.joblib")
        self.logger.info("StandardScaler fitted and saved.")

    def transform(self, df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """Transform the numerical columns using the fitted StandardScaler.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            numerical_columns (List[str]): List of numerical column names to scale.
            
        Returns:
            pd.DataFrame: The transformed dataframe."""
        if self.scaler is None:
            self.logger.error("Scaler has not been fitted yet.")
            raise RuntimeError("Scaler has not been fitted yet.")
        
        scaled_array = self.scaler.transform(df[numerical_columns])
        scaled_df = pd.DataFrame(scaled_array, columns=numerical_columns, index=df.index)
        
        df.update(scaled_df)
        return df
    
    def fit_transform(self, df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """Fit and transform the numerical columns using StandardScaler.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            numerical_columns (List[str]): List of numerical column names to scale.
        returns:
            pd.DataFrame: The transformed dataframe.
        """
        self.fit(df, numerical_columns)
        return self.transform(df, numerical_columns)
    
    def save_scaler(self, filepath: str) -> None:
        """Save the fitted scaler to a joblib file and log as mlflow artifact.
        
        Args:
            filepath (str): The path to save the scaler.
        """
        joblib.dump(self.scaler, filepath)
        mlflow.log_artifact(filepath, artifact_path="feature_scaling")
        self