# Load trained model from mlflow


from utils import LoggerMixin

class LoadModel(LoggerMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("LoadData",config, "logging")
        self.model = None 

    def load_model(self):
        """Load the best model from mlflow for evaluation
        
        Returns:
            A model instance
        """
        self.logger.info("="*50)
        self.logger.info("LOADING MODEL FROM MLFLOW")
        self.logger.info("="*50)
        try:
            if self.model is None:
                self.model = self.config.get('model_uri','models:/RandomForest_model@production')
        
        except Exception as e:
            self.logger.error(f'Error loading model from mlflow: {e}')
            raise