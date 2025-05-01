from collections import OrderedDict
from typing import override

import numpy as np
import heapq
from predictors import Predictor
from recommenders.plain import PlainRecommender


class ScoreBoostRecommender(PlainRecommender):
    """
    Recommend items based on boosted score of items.
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
        scores = super()._predict_base_scores(user)
        bid_scores = []
        for bid in self.payments:
            idx, val = bid
            if np.ma.is_masked(scores[idx]):
                continue
            # Make score negative for max-heap implementation
            heapq.heappush(bid_scores, (-self._boost(scores[idx], val), idx))
        res = [None] * count
        i = 0
        slot_set = set(self.promotion_slots)
        scores_order = np.ma.argsort(scores, fill_value=-1)[::-1]
        chosen_items = set()
        for idx in range(count):
            if idx not in slot_set:
                while scores_order[i] in chosen_items:
                    i += 1
                res[idx] = (scores_order[i], scores[scores_order[i]])
                if bid_scores[0][1] == idx:
                    heapq.heappop(bid_scores)
                i += 1
            else:
                res[idx] = (bid_scores[0][1], -bid_scores[0][0])
                heapq.heappop(bid_scores)
            chosen_items.add(res[idx][0])
        return res
