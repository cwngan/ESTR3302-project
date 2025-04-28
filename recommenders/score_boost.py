from typing import override

import numpy as np
from predictors import Predictor
from recommenders import Recommender


class ScoreBoostRecommender(Recommender):
    """
    Recommend items based on boosted score of items.
    """

    def __init__(
        self,
        predictor: Predictor,
        users: int,
        items: int,
        bids: list[tuple[int, float]],
        promotion_slots: list[bool],
        alpha: float = 0.1,
        beta: float = 50,
    ):
        self.predictor = predictor
        self.users = users
        self.items = items
        self.bids = bids
        self.alpha = alpha
        self.beta = beta
        self.promotion_slots = promotion_slots

    def _boost(self, orig: float, val: float):
        return orig + self.alpha * np.log1p(val * self.beta)

    @override
    def recommend_items(self, user, count):
        scores = self.predictor.predict(
            entries=[(user, i) for i in range(self.items)], quiet=True
        )
        mask = np.zeros(shape=scores.shape, dtype=bool)
        for bid in self.bids:
            idx, val = bid
            scores[idx] = self._boost(scores[idx], val)
            mask[idx] = True
        plain_order = np.ma.argsort(
            np.ma.masked_array(scores, mask=mask), fill_value=-999999
        )[::-1]
        promotion_order = np.ma.argsort(
            np.ma.masked_array(scores, mask=~mask), fill_value=-999999
        )[::-1]
        res = np.zeros((count,), dtype=int)
        i = 0
        j = 0
        for idx in range(count):
            if self.promotion_slots[idx]:
                res[idx] = promotion_order[j]
                j += 1
            else:
                res[idx] = plain_order[i]
                i += 1
        return res
