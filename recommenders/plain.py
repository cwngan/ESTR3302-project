from typing import override

import numpy as np
from predictors import Predictor
from recommenders import Recommender


class PlainRecommender(Recommender):
    """
    Recommend items based on plain rating of items.
    """

    def __init__(
        self,
        predictor: Predictor,
        users: int,
        items: int,
    ):
        self.predictor = predictor
        self.users = users
        self.items = items

    def _predict_base_ratings(self, user):
        ratings = self.predictor.predict(
            entries=[(user, i) for i in range(self.items)], quiet=True
        )
        ratings = np.ma.masked_less(ratings, 3)
        return ratings

    @override
    def recommend_items(self, user, count):
        ratings = self._predict_base_ratings(user)
        res_order = np.ma.argsort(ratings, fill_value=-1)[::-1][:count]
        res_rating = np.ma.sort(ratings, fill_value=-1)[::-1][:count]
        return list(zip(res_order, res_rating, res_rating))
