"""Hyperparameter using optuna"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from utils import LoggerMixin
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_slice, plot_param_importances

class HyperParameterTuner(LoggerMixin):
    """
    Hyperparameter tuning with Optuna.
    
    Attributes:
        config: Configuration dictionary
        study: Optuna study object
        best_params: Best hyperparameters found
        best_score: Best score achieved
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("HyperParameterTuner",config,"logging")

        self.study = None
        self.best_params = None
        self.best_score = None 

    def suggest_params(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            model_name: Name of the model
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        search_space = self.config['hyperparameter_tuning']['search_spaces'].get(model_name, {})
        params = {}

        

