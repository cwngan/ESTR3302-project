import numpy as np
from abc import ABC, abstractmethod


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

    def predict_all(self) -> list[float]:
        """
        Predict the rating for all users and items.
        """
        raise NotImplementedError("Subclasses should implement this method.")
