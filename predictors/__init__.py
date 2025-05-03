from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix


class Predictor(ABC):
    """
    Abstract base class for all predictors.
    """

    shape: tuple[int, int]

    @abstractmethod
    def predict(self, entries: list[tuple[int, int]], quiet: bool) -> np.ndarray:
        """
        Predict the rating for given users and items.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def predict_all(self, quiet: bool) -> np.ndarray:
        """
        Predict the rating for all users and items.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def train(self, training_data: csr_matrix):
        """
        Train the predictor.
        """
        raise NotImplementedError("Subclasses should implement this method.")
