from abc import ABC, abstractmethod

from predictors import Predictor


class Recommender(ABC):
    """
    Abstract base class for all recommenders.
    """

    predictor: Predictor
    users: int
    items: int

    @abstractmethod
    def recommend_items(self, user: int, count: int) -> list[int]:
        """
        Recommend items for a given user.
        """
        raise NotImplementedError("Subclasses should implement this method.")
