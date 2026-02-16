"""configuration management for FastAPI"""
import yaml
from pathlib import Path
from typing import Optional
import mlflow 

class Settings:
    """Application Settings"""
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_mlflow()

    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, "r") as file:
                return yaml.safe_load(file)
            return self._get_default_config()
        
    def _get_default_config(self) -> dict:
        """Default configuration if no file exists"""
        return { 
            "mlflow_tracking_uri" : "sqlite:///mlflow_car.db",
            "model_uri" : "RandomForest_model@production",
            "api" : {
                "title" : "Car Price Prediction API",
                "description" : "Predict car prices using machine learning",
                "version" : "1.0.0",
                "host" : "0.0.0.0",
                "port" : 8000
            }
        }
    
    def _setup_mlflow(self):
        """Setup MLflow tracking URI"""
        tracking_uri = self.config.get("mlflow_tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    @property
    def api_title(self) -> str:
        return self.config.get('api', {}).get("title", "Car Price Prediction API")
    
    @property
    def api_description(self) -> str:
        return self.config.get("api", {}).get("description", "Predict car prices using machine learning")
    
    @property
    def api_version(self) -> str:
        return self.config.get("api", {}).get("version", "1.0.0")
    
settings = Settings()