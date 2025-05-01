from typing import override

import numpy as np
from predictors import Predictor
from recommenders import Recommender


class PlainRecommender(Recommender):
    """
    Recommend items based on boosted score of items.
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

    def _predict_base_scores(self, user):
        scores = self.predictor.predict(
            entries=[(user, i) for i in range(self.items)], quiet=True
        )
        scores = np.ma.masked_less(scores, 3)
        return scores

    @override
    def recommend_items(self, user, count):
        scores = self._predict_base_scores(user)
        res_order = np.ma.argsort(scores, fill_value=-1)[::-1][:count]
        res_score = np.ma.sort(scores, fill_value=-1)[::-1][:count]
        return zip(res_order, res_score)
