"""
Base recommender class for investment recommendations
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRecommender(ABC):
    """Abstract base class for recommendation systems"""
    
    @abstractmethod
    def train(self, training_data: Any) -> None:
        """Train the recommendation model"""
        pass
    
    @abstractmethod
    def predict(self, user_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on user features"""
        pass
    
    @abstractmethod
    def update(self, feedback_data: Any) -> None:
        """Update the model with new feedback data"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model to disk"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load the model from disk"""
        pass 