from abc import ABC, abstractmethod
import numpy as np


class Predictor(ABC):
    """
    Abstract base class for all predictors.
    """

    training_data: np.ndarray

    @abstractmethod
    def predict(self, entries: list[tuple[int]]) -> list[float]:
        """
        Predict the rating for given users and items.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def predict_all(self) -> np.ndarray:
        """
        Predict the rating for all users and items.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def train(self):
        """
        Train the predictor.
        """
        raise NotImplementedError("Subclasses should implement this method.")
