"""  
Core model training module with sklearn models
"""

import pandas as pd
import numpy as np
import time
from utils import LoggerMixin
from typing import Dict, Any, List, Optional
from sklearn.linear_model import Ridge 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score

class ModelTrainer(LoggerMixin):
    """Train regression models with cross-validation
    
    Attributes:
        config: Configuration dictionary
        model: Trained model instance
        model_name: Name of the model
        training_time: Time taken to train
        cv_scores: Cross-validation scores
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config 
        self.logger = self.setup_class_logger("ModelTrainer", config, "logging")
        self.model = None 
        self.model_name = None
        self.training_time = 0.0

    def train_baseline_models(self, model_name):
        """Train all baseline models"""
        learning_algorithms = self.config.get('learning_algorithms',[])

        if model_name not in learning_algorithms:
            raise ValueError(f"Unkown model: {model_name}")

    def _get_model_and_params(self, model_name: str, params: Dict[str, Any]):
        """Get model instance based on names
        
        Args:
            model_name: Name of the model
            params: Model's hyperparameters
            
        Returns:
            Model instance
        """
        learning_alogrithms = self.config.get('learning_algorithms',[])

        if model_name not in learning_alogrithms:
            raise ValueError(f'Unknown model: {model_name}')
        
        model_map = {
            'Ridge' : Ridge,
            'RandomForest' : RandomForestRegressor,
            'XGBoost' : XGBRegressor,
            'LightGBM' : LGBMRegressor
        }
        
        return model_map[model_name](**params)
    

    def train(
            self,
            X_train: np.array,
            y_train: np.array,
            model_name: str
            ):
        """        
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            params: Model hyperparameters (optional)
            
        Returns:
            Trained model
        """

        self.logger.info(f"\nTraining model {model_name}...")
        self.logger.info(f'-'*60)

        self.model_name = model_name

        # Get model parameters
        params = self.config['baseline_models']['models'][model_name].get('params',{})

        self.logger.info(f'Parameters for {model_name}: {params}')

        # initialize model
        self.model = self._get_model_instance(model_name,params)

        # train model
        start_time = time.time()
        self.model.fit(X_train,y_train)
        self.training_time = time.time() - start_time

        self.logger.info(f'Training completed in {self.training_time}')
        return self.model
    
    def regression_metrics(self):
        """Regression metrics to evaluate model"""

    
    def evaluate_model(self, model_name):