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
        cv_scores = None

    def _get_model_instance(self, model_name: str, params: Dict[str, Any]):
        """Get model instance based on names
        
        Args:
            model_name: Name of the model
            params: Model's hyperparameters
            
        Returns:
            Model instance
        """
        model_map = {
            'Ridge' : Ridge,
            'RandomForest' : RandomForestRegressor,
            'XGBoost' : XGBRegressor,
            'LightGBM' : LGBMRegressor
        }

        if model_name not in model_map:
            raise ValueError(f'Unkown model: {model_name}')
        
        return model_map[model_name](**params)
    

    def train(
            self,
            X_train: np.array,
            y_train: np.array,
            model_name: str,
            params: Optional[Dict[str, Any]] = None
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
        if params is None:
            if model_name == "Ridge":
                params = self.config['models']['baseline']['params']
                
            else:
                params = self.config['models']['tree_based'][model_name]['params']

        self.logger.info(f'Parameters: {params}')

        # initialize model
        self.model = self._get_model_instance(model_name,params)

        # train model
        start_time = time.time()
        self.model.fit(X_train,y_train)
        self.training_time = time.time() - start_time

        self.logger.info(f'Training completed in {self.training_time}')
        return self.model
    
    def cross_validate(
            self,
            X_train: np.array,
            y_train: np.array
    ) -> Dict[str, float]:
        """       
        Perform cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV scores
        """
        if not self.config['cross_validation']['enabled']:
            self.logger.info("Cross validation disabled. Skipping...")
            return {}
        
        scoring = self.config['cross_validation'].get('scoring','neg_mean_squared_error')
        
        cv_scores = cross_val_score(
            X_train, y_train,scoring=scoring, cv=KFold(10), n_jobs=-1 
        )

        self.cv_scores = -cv_scores
        
        cv_results = {
            'cv_mean' : self.cv_scores.mean(),
            'cv_std' : self.cv_scores.std(),
            'cv_min' : self.cv_scores.min(),
            'cv_max' : self.cv_scores.max(),
            'cv_scores' : self.cv_scores.tolist()
        }

        self.logger.info(f"CV {scoring}: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
        self.logger.info(f"CV range: [{cv_results['cv_min']:.4f}, {cv_results['cv_max']:.4f}]")

    def get_model(self):
        "Get trained model"
        self.model