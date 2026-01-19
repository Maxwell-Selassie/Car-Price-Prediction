"""Feature encoding utilities for preprocessing.

Workflow:
1. Import necessary libraries.
2. Use scikit-learn "OneHotEncoder" for categorical encoding.
3. Define fit and transform functions.
4. Save fitted encoder as a joblib file
5. Log artifacts using mlflow.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import List, Dict, Any 
import numpy as np
from utils import LoggerMixin
import joblib
import mlflow

class FeatureEncoder(LoggerMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("FeatureEncoder", config, "logging")
        self.encoder = OneHotEncoder | None = None

    def fit(self, df: pd.DataFrame, categorical_columns: List[str]) -> None:
        """Fit the OneHotEncoder on the categorical columns.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            categorical_columns (List[str]): List of categorical column names to encode.
            
        Returns:
            None
        """
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(df[categorical_columns])
        self.save_encoder(self.config['encoding']['artifact_dir'] + "artifacts/one_hot_encoder.joblib")
        self.logger.info("OneHotEncoder fitted and saved.")

    def transform(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Transform the categorical columns using the fitted OneHotEncoder.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            categorical_columns (List[str]): List of categorical column names to encode.
            
        Returns:
            pd.DataFrame: The transformed dataframe."""
        if self.encoder is None:
            self.logger.error("Encoder has not been fitted yet.")
            raise RuntimeError("Encoder has not been fitted yet.")
        
        encoded_array = self.encoder.transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out(categorical_columns), index=df.index)
        
        df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
        return df
    
    def fit_transform(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Fit and transform the categorical columns using OneHotEncoder.
        Args:
            df (pd.DataFrame): The input dataframe.
            categorical_columns (List[str]): List of categorical column names to encode.
        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        self.fit(df, categorical_columns)
        return self.transform(df, categorical_columns)

    def save_encoder(self, file_path: str):
        """Save the fitted encoder to a joblib file and log it as an MLflow artifact."""
        joblib.dump(self.encoder, file_path)
        mlflow.log_artifact(file_path, artifact_path="feature_encoding")

        self.logger.info(f"Encoder saved and logged to MLflow at {file_path}")

