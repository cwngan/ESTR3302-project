from typing import override

import numpy as np
from predictors import Predictor
from recommenders.plain import PlainRecommender


class AuctionRecommender(PlainRecommender):
    """
    Recommend items based on auction scores.
    """

    def __init__(
        self,
        predictor: Predictor,
        users: int,
        items: int,
        bids: list[tuple[int, int, float]],
        promotion_slots: list[int],
        alpha: float = 0.1,
        beta: float = 50,
    ):
        super().__init__(predictor, users, items)
        self.bids = bids
        self.alpha = alpha
        self.beta = beta
        self.promotion_slots = promotion_slots

    def _boost(self, orig: float, val: float):
        return orig + self.alpha * np.log1p(val * self.beta)

    @override
    def recommend_items(self, user, count):
        scores = super()._predict_base_scores(user)
        plain_order = np.ma.argsort(scores, fill_value=-1)[::-1]
        plain_scores = np.ma.sort(scores, fill_value=-1)[::-1]
        rank_map = {video: rank for rank, video in enumerate(plain_order)}
        bids_by_slot: list[list[tuple[int, int, float]]] = [
            [] for _ in self.promotion_slots
        ]
        for bid in self.bids:
            bids_by_slot[bid[1]].append(bid)
        auction_winners: set[int] = set()
        res = [None] * count
        for slot_idx, slot in enumerate(bids_by_slot):
            winner = -1
            winner_score = -1
            for bid in slot:
                idx, _, val = bid
                if np.ma.is_masked(scores[idx]):
                    continue
                if rank_map[idx] < self.promotion_slots[slot_idx]:
                    continue
                boosted_score = self._boost(scores[idx], val)
                if boosted_score > winner_score:
                    winner = idx
                    winner_score = boosted_score
            if winner != -1:
                auction_winners.add(winner)
                res[self.promotion_slots[slot_idx]] = (winner, winner_score)
        j = 0
        for i in range(count):
            if res[i] is not None:
                continue
            while plain_order[j] in auction_winners:
                j += 1
            res[i] = (plain_order[j], plain_scores[j])
            j += 1
        return res
