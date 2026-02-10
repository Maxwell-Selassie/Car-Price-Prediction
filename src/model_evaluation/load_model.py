import mlflow
from utils import LoggerMixin

class LoadModel(LoggerMixin):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("LoadData", config, "logging")
        
        # ADD THIS LINE - Set tracking URI BEFORE loading model
        tracking_uri = self.config.get('mlflow_tracking_uri')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            self.logger.info(f"MLflow Tracking URI set to: {tracking_uri}")
        
        self.model = None

    def load_model(self):
        """Load the best model from mlflow for evaluation
        
        Returns:
            A model instance
        """
        
        try:
            if self.model is None:
                model_uri = self.config.get('model_uri','models:/RandomForest_model@production')
                self.model = mlflow.pyfunc.load_model(model_uri=model_uri)
                self.logger.info(f"✓ Model loaded successfully from: {model_uri}")
                
        except Exception as e:
            self.logger.error(f'Error loading model from mlflow: {e}')
            raise
            
        return self.model