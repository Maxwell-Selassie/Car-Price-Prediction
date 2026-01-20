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
from typing import List, Dict, Any, Optional
from pathlib import Path
from utils import LoggerMixin
import shutil
import joblib
import mlflow

class FeatureEncoder(LoggerMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("FeatureEncoder", config, "logging")
        self.encoder : Optional[OneHotEncoder] = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the OneHotEncoder on the categorical columns.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            None
        """
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(df[categorical_columns])
        self.save_encoder()
        self.logger.info("OneHotEncoder fitted and saved.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the categorical columns using the fitted OneHotEncoder.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: The transformed dataframe."""
        if self.encoder is None:
            self.logger.error("Encoder has not been fitted yet.")
            raise RuntimeError("Encoder has not been fitted yet.")
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        missing_cols = [col for col in categorical_columns if col not in df.columns]
        if missing_cols:
            self.logger.error(f"The following columns are missing in the DataFrame for encoding: {missing_cols}")
            raise ValueError(f"The following columns are missing in the DataFrame for encoding: {missing_cols}")
        
        encoded_array = self.encoder.transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out(categorical_columns), index=df.index)
        
        df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the categorical columns using OneHotEncoder.
        Args:
            df (pd.DataFrame): The input dataframe.
        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        self.fit(df)
        return self.transform(df)

    def save_encoder(self, version: int = 1):
        """Save the fitted encoder to a joblib file and log it as an MLflow artifact."""
        output_dir = self.config['save_artifacts'].get('encoder_path', 'artifacts/encoders')
        file_path = Path(f"{output_dir}/one_hot_encoder_v{version}.joblib")
        
        if file_path.is_dir():
            shutil.rmtree(file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.encoder, file_path)
        mlflow.log_artifact(file_path)

        self.logger.info(f"Encoder saved and logged to MLflow at {file_path}")

