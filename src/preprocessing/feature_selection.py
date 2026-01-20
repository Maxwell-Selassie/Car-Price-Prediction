"""Feature selection utilities for preprocessing.
Workflow:
1. Import necessary libraries.
2. Use Recursive Feature Elimination (RFE) for feature selection.
4. Define fit and transform functions.
5. Save fitted selector as a joblib file
6. Save selected features list as a json file
6. Log artifacts using mlflow."""

import pandas as pd
from sklearn.feature_selection import RFECV
from typing import List, Dict, Any
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from utils import LoggerMixin, ensure_directory
import joblib
import json
import mlflow

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
        if self.selector is None:
            self.logger.error("Selector has not been initialized.")
            raise RuntimeError("Selector has not been initialized.")
        
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

    def save_selector(self) -> None:
        """Save the fitted selector to a joblib file.
        
        Args:
            path (str): The path to save the selector file.
        """
        path = self.config['save_artifacts'].get('selection_path', 'artifacts/feature_selection/rfecv_selector.joblib')
        ensure_directory(path)
        joblib.dump(self.selector, path)
        mlflow.log_artifact(path)
        self.logger.info(f"Feature selector saved at {path}.")

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit the selector and transform the input features dataframe.
        
        Args:
            X (pd.DataFrame): The input features dataframe.
            y (pd.Series): The target variable series.
        """
        self.fit(X, y)
        return self.transform(X)
        
    def selected_features(self, X: pd.DataFrame) -> List[str]:
        """Get the list of selected feature names after fitting.
        
        Args:
            X (pd.DataFrame): The input features dataframe.
        Returns:
            List[str]: The list of selected feature names.
        """
        if self.selector is None:
            self.logger.error("Selector has not been fitted yet.")
            raise RuntimeError("Selector has not been fitted yet.")
        
        selected_feature_names = X.columns[self.selector.support_]
        feature_ranking = pd.Series(self.selector.ranking_, index=X.columns).sort_values()
        mlflow.log_param('optimal_number_of_features', self.selector.n_features_)
        self.logger.info(f"Selected features: {selected_feature_names.tolist()}")
        self.logger.info(f"Feature ranking:\n{feature_ranking}")
        return selected_feature_names
    
    def save_selected_features(self, X) -> None:
        """Save the list of selected feature names to a json file.
        
        Args:
            X (pd.DataFrame): The input features dataframe.
        """
        selected_feature_names = self.extract_selected_feature_names(X)
        path = self.config['save_artifacts'].get('selected_features_path', 'artifacts/feature_selection/selected_features.json')
        ensure_directory(path)
        with open(path, 'w') as f:
            json.dump(selected_feature_names, f)
        mlflow.log_artifact(path)
        self.logger.info(f"Selected features saved at {path}.")