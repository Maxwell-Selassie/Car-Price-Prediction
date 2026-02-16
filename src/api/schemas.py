"""Pydantic schemas for request/response validation"""
"""Pydantic schemas for request/response validation"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class CarFeatures(BaseModel):
    """Input features for car price prediction"""
    
    # Numeric features
    mileage: float = Field(..., ge=0, description="Mileage (kmpl or km/kg)")
    seats: float = Field(..., ge=2, le=10, description="Number of seats")
    age: int = Field(..., ge=0, le=100, description="Age of the car in years")
    km_driven_log: float = Field(..., description="Log-transformed kilometers driven")
    max_power_log: float = Field(..., description="Log-transformed maximum power")
    engine_sqrt: float = Field(..., ge=0, description="Square root of engine capacity")
    
    # Categorical features - Fuel (one-hot encoded)
    fuel_Diesel: int = Field(0, ge=0, le=1, description="Is fuel type Diesel? (0 or 1)")
    fuel_LPG: int = Field(0, ge=0, le=1, description="Is fuel type LPG? (0 or 1)")
    fuel_Petrol: int = Field(0, ge=0, le=1, description="Is fuel type Petrol? (0 or 1)")
    
    # Categorical features - Seller Type (one-hot encoded)
    seller_type_Dealer: int = Field(0, ge=0, le=1, description="Is seller type Dealer? (0 or 1)")
    seller_type_Individual: int = Field(0, ge=0, le=1, description="Is seller type Individual? (0 or 1)")
    
    # Categorical features - Transmission (one-hot encoded)
    transmission_Automatic: int = Field(0, ge=0, le=1, description="Is transmission Automatic? (0 or 1)")
    transmission_Manual: int = Field(0, ge=0, le=1, description="Is transmission Manual? (0 or 1)")
    
    # Categorical features - Owner (one-hot encoded)
    owner_First_Owner: int = Field(0, ge=0, le=1, description="Is owner First Owner? (0 or 1)")
    owner_Fourth_Above_Owner: int = Field(0, ge=0, le=1, description="Is owner Fourth & Above Owner? (0 or 1)")
    owner_Second_Owner: int = Field(0, ge=0, le=1, description="Is owner Second Owner? (0 or 1)")
    owner_Third_Owner: int = Field(0, ge=0, le=1, description="Is owner Third Owner? (0 or 1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mileage": 18.9,
                "seats": 5.0,
                "age": 7,
                "km_driven_log": 9.615805,
                "max_power_log": 4.484225,
                "engine_sqrt": 34.6,
                "fuel_Diesel": 0,
                "fuel_LPG": 0,
                "fuel_Petrol": 1,
                "seller_type_Dealer": 0,
                "seller_type_Individual": 1,
                "transmission_Automatic": 0,
                "transmission_Manual": 1,
                "owner_First_Owner": 1,
                "owner_Fourth_Above_Owner": 0,
                "owner_Second_Owner": 0,
                "owner_Third_Owner": 0
            }
        }
    
    @validator('fuel_Diesel', 'fuel_LPG', 'fuel_Petrol')
    def validate_fuel_encoding(cls, v, values):
        """Ensure only one fuel type is selected"""
        if v not in [0, 1]:
            raise ValueError("Fuel encoding must be 0 or 1")
        return v
    
    @validator('seller_type_Dealer', 'seller_type_Individual')
    def validate_seller_encoding(cls, v):
        """Ensure seller type encoding is valid"""
        if v not in [0, 1]:
            raise ValueError("Seller type encoding must be 0 or 1")
        return v
    
    @validator('transmission_Automatic', 'transmission_Manual')
    def validate_transmission_encoding(cls, v):
        """Ensure transmission encoding is valid"""
        if v not in [0, 1]:
            raise ValueError("Transmission encoding must be 0 or 1")
        return v
    
    @validator('owner_First_Owner', 'owner_Fourth_Above_Owner', 'owner_Second_Owner', 'owner_Third_Owner')
    def validate_owner_encoding(cls, v):
        """Ensure owner encoding is valid"""
        if v not in [0, 1]:
            raise ValueError("Owner encoding must be 0 or 1")
        return v


class CarFeaturesSimple(BaseModel):
    """Simplified input that will be transformed to CarFeatures"""
    
    # Raw inputs from user
    year: int = Field(..., ge=1900, le=2030, description="Manufacturing year")
    km_driven: float = Field(..., ge=0, description="Kilometers driven")
    fuel: str = Field(..., description="Fuel type (Petrol/Diesel/LPG/CNG/Electric)")
    seller_type: str = Field(..., description="Seller type (Individual/Dealer/Trustmark Dealer)")
    transmission: str = Field(..., description="Transmission type (Manual/Automatic)")
    owner: str = Field(..., description="Owner type (First Owner/Second Owner/Third Owner/Fourth & Above Owner)")
    mileage: float = Field(..., ge=0, description="Mileage (kmpl or km/kg)")
    engine: float = Field(..., ge=0, description="Engine capacity (CC)")
    max_power: float = Field(..., ge=0, description="Maximum power (bhp)")
    seats: float = Field(..., ge=2, le=10, description="Number of seats")
    
    class Config:
        json_schema_extra = {
            "example": {
                "year": 2019,
                "km_driven": 15000,
                "fuel": "Petrol",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "First Owner",
                "mileage": 18.9,
                "engine": 1197,
                "max_power": 88.5,
                "seats": 5
            }
        }
    
    @validator('fuel')
    def validate_fuel(cls, v):
        allowed = ['Petrol', 'Diesel', 'LPG', 'CNG', 'Electric']
        if v not in allowed:
            raise ValueError(f'Fuel must be one of {allowed}')
        return v
    
    @validator('transmission')
    def validate_transmission(cls, v):
        allowed = ['Manual', 'Automatic']
        if v not in allowed:
            raise ValueError(f'Transmission must be one of {allowed}')
        return v
    
    @validator('owner')
    def validate_owner(cls, v):
        allowed = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
        if v not in allowed:
            raise ValueError(f'Owner must be one of {allowed}')
        return v


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    
    predicted_price: float = Field(..., description="Predicted car price")
    currency: str = Field(default="INR", description="Currency of prediction")
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_interval: Optional[dict] = Field(None, description="Confidence interval if available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 450000.50,
                "currency": "INR",
                "model_version": "RandomForest_model@production:3",
                "prediction_timestamp": "2026-02-10T10:30:00",
                "confidence_interval": None
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    cars: List[CarFeaturesSimple] = Field(..., min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "cars": [
                    {
                        "year": 2019,
                        "km_driven": 15000,
                        "fuel": "Petrol",
                        "seller_type": "Individual",
                        "transmission": "Manual",
                        "owner": "First Owner",
                        "mileage": 18.9,
                        "engine": 1197,
                        "max_power": 88.5,
                        "seats": 5
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse]
    total_predictions: int
    batch_timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    mlflow_tracking_uri: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    model_version: str
    model_uri: str
    model_type: str
    feature_names: List[str]
    n_features: int