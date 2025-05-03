from typing import override

import numpy as np
import heapq
from predictors import Predictor
from recommenders.plain import PlainRecommender


class RatingBoostRecommender(PlainRecommender):
    """
    Recommend items based on boosted rating of items.
    """

    def __init__(
        self,
        predictor: Predictor,
        users: int,
        items: int,
        payments: list[tuple[int, float]],
        promotion_slots: list[int],
        alpha: float = 0.1,
        beta: float = 50,
    ):
        super().__init__(predictor, users, items)
        self.payments = payments
        self.alpha = alpha
        self.beta = beta
        self.promotion_slots = promotion_slots

    def _boost(self, orig: float, val: float):
        return orig + self.alpha * np.log1p(val * self.beta)

    @override
    def recommend_items(self, user, count):
        ratings = super()._predict_base_ratings(user)
        bid_ratings = []
        for bid in self.payments:
            idx, val = bid
            if np.ma.is_masked(ratings[idx]):
                continue
            # Make score negative for max-heap implementation
            heapq.heappush(bid_ratings, (-self._boost(ratings[idx], val), idx))
        res = [None] * count
        i = 0
        slot_set = set(self.promotion_slots)
        ratings_order = np.ma.argsort(ratings, fill_value=-1)[::-1]
        chosen_items = set()
        for idx in range(count):
            if idx not in slot_set:
                while ratings_order[i] in chosen_items:
                    i += 1
                res[idx] = (
                    ratings_order[i],
                    ratings[ratings_order[i]],
                    ratings[ratings_order[i]],
                )
                if bid_ratings[0][1] == idx:
                    heapq.heappop(bid_ratings)
                i += 1
            else:
                res[idx] = (
                    bid_ratings[0][1],
                    -bid_ratings[0][0],
                    ratings[bid_ratings[0][1]],
                )
                heapq.heappop(bid_ratings)
            chosen_items.add(res[idx][0])
        return res
