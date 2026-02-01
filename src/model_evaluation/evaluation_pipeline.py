"""Evaluation pipeline Module"""

import pandas as pd
import numpy as np
from utils import LoggerMixin
from model_evaluation import LoadData, LoadModel, RegressionMetrics,RunPredictions
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

class ModelEvaluation(LoggerMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.logger = self.setup_class_logger("ModelEvaluation",config,"logging")
        