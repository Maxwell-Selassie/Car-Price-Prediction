"""Load test set for model evaluation"""

import pandas as pd
from pathlib import Path
from utils import LoggerMixin

class LoadData(LoggerMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("LoadTestData", config, "logging")
        self.X_test = None 
        self.y_test = None 

    def load_test_data(self):
        """Load the held-out test set for evaluation"""
        self.logger.info("="*50)
        self.logger.info(f"LOADING TEST DATA")
        self.logger.info("="*50)

        # target variable
        target_variable = self.config.get("target_column", "selling_price_log")
        
        try: 
            test_df_path = self.config.get("test_file_path", "data/processed/test_processed_v1.csv")
            test_df = pd.read_csv(test_df_path)
            if target_variable not in test_df.columns:
                raise ValueError(f"Target varaible must be present in dataset")
            
            self.X_test = test_df.drop(columns=[target_variable]).copy()
            self.y_test = test_df[target_variable]

            self.logger.info(f"Number of features: {self.X_test.shape[1]}")
            self.logger.info(f'Shape of test fetures: {self.X_test.shape}')
            self.logger.info(f"Target variable shape: {self.y_test.shape}")
            self.logger.info("Success: Data Loaded successfully")
            
        except Exception as e:
            self.logger.error(f'Error loading test data: {e}')