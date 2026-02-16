"""Model loading and management"""
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ML model loading and predictions"""
    
    def __init__(self):
        self.model = None
        self.model_info: Dict = {}
        self.feature_names: List[str] = []
        self._load_model()
    
    def _load_model(self):
        """Load model from MLflow"""
        try:
            logger.info(f"Loading model from: {settings.model_uri}")
            
            # Load sklearn model directly
            self.model = mlflow.sklearn.load_model(model_uri=settings.model_uri)
            
            # Get model metadata
            from mlflow import MlflowClient
            client = MlflowClient()
            
            # Parse model URI
            uri_parts = settings.model_uri.replace("models:/", "").split("@")
            model_name = uri_parts[0]
            alias = uri_parts[1] if len(uri_parts) > 1 else "production"
            
            # Get model version info
            model_version = client.get_model_version_by_alias(model_name, alias)
            
            self.model_info = {
                'name': model_name,
                'version': model_version.version,
                'alias': alias,
                'uri': settings.model_uri,
                'run_id': model_version.run_id
            }
            
            # Set feature names based on your selected features
            self.feature_names = [
                "mileage",
                "seats",
                "age",
                "km_driven_log",
                "max_power_log",
                "engine_sqrt",
                "fuel_Diesel",
                "fuel_LPG",
                "fuel_Petrol",
                "seller_type_Dealer",
                "seller_type_Individual",
                "transmission_Automatic",
                "transmission_Manual",
                "owner_First Owner",
                "owner_Fourth & Above Owner",
                "owner_Second Owner",
                "owner_Third Owner"
            ]
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Model: {model_name}")
            logger.info(f"  Version: {model_version.version}")
            logger.info(f"  Features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def transform_features(self, car_features: Dict) -> Dict:
        """Transform raw features to model features with engineering"""
        from datetime import datetime
        
        # Calculate age from year
        current_year = datetime.now().year
        age = current_year - car_features['year']
        
        # Log transformations
        km_driven_log = np.log1p(car_features['km_driven'])  # log(1 + x) to handle 0
        max_power_log = np.log1p(car_features['max_power'])
        
        # Square root transformation
        engine_sqrt = np.sqrt(car_features['engine'])
        
        # One-hot encoding for fuel
        fuel_Diesel = 1 if car_features['fuel'] == 'Diesel' else 0
        fuel_LPG = 1 if car_features['fuel'] == 'LPG' else 0
        fuel_Petrol = 1 if car_features['fuel'] == 'Petrol' else 0
        
        # One-hot encoding for seller_type
        seller_type_Dealer = 1 if car_features['seller_type'] == 'Dealer' else 0
        seller_type_Individual = 1 if car_features['seller_type'] == 'Individual' else 0
        
        # One-hot encoding for transmission
        transmission_Automatic = 1 if car_features['transmission'] == 'Automatic' else 0
        transmission_Manual = 1 if car_features['transmission'] == 'Manual' else 0
        
        # One-hot encoding for owner
        owner_First_Owner = 1 if car_features['owner'] == 'First Owner' else 0
        owner_Fourth_Above_Owner = 1 if car_features['owner'] == 'Fourth & Above Owner' else 0
        owner_Second_Owner = 1 if car_features['owner'] == 'Second Owner' else 0
        owner_Third_Owner = 1 if car_features['owner'] == 'Third Owner' else 0
        
        return {
            'mileage': car_features['mileage'],
            'seats': car_features['seats'],
            'age': age,
            'km_driven_log': km_driven_log,
            'max_power_log': max_power_log,
            'engine_sqrt': engine_sqrt,
            'fuel_Diesel': fuel_Diesel,
            'fuel_LPG': fuel_LPG,
            'fuel_Petrol': fuel_Petrol,
            'seller_type_Dealer': seller_type_Dealer,
            'seller_type_Individual': seller_type_Individual,
            'transmission_Automatic': transmission_Automatic,
            'transmission_Manual': transmission_Manual,
            'owner_First Owner': owner_First_Owner,
            'owner_Fourth & Above Owner': owner_Fourth_Above_Owner,
            'owner_Second Owner': owner_Second_Owner,
            'owner_Third Owner': owner_Third_Owner
        }
    
    def preprocess_input(self, car_features: Dict) -> pd.DataFrame:
        """Preprocess input features to match model expectations"""
        
        # Transform features
        transformed = self.transform_features(car_features)
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([transformed], columns=self.feature_names)
        
        return input_df
    
    def predict(self, car_features: Dict) -> float:
        """Make a single prediction"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess input
        input_df = self.preprocess_input(car_features)
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        
        return float(prediction)
    
    def predict_batch(self, cars_features: List[Dict]) -> List[float]:
        """Make batch predictions"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess all inputs
        input_dfs = [self.preprocess_input(car) for car in cars_features]
        combined_df = pd.concat(input_dfs, ignore_index=True)
        
        # Make predictions
        predictions = self.model.predict(combined_df)
        
        return [float(pred) for pred in predictions]
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return self.model_info.copy()

# Global model manager instance
model_manager = ModelManager()