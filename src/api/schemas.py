"""Pydantic schemas for request/response validation"""
from pydantic import BaseModel, Field, validator 
from typing import Optional, List 
from datetime import datetime 

class CarFeatures(BaseModel):
    """Input features for car price prediction"""

    