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

    @override
    def recommend_items(self, user, count):
        scores = self.predictor.predict(
            entries=[(user, i) for i in range(self.items)], quiet=True
        )
        res = np.argsort(scores)[::-1][:count]
        print(np.sort(scores)[::-1][:count])
        return res
