from functools import partial
from typing import override
import numpy as np
from predictors import Predictor


class LatentFactorPredictor(Predictor):
    """
    A latent factor predictor trained with given training data.
    """

    k: int
    p: np.ndarray
    q: np.ndarray
    lmda: float
    prediction: np.ndarray = None

    def __init__(
        self,
        training_data: np.ndarray,
        k: int,
        p: np.ndarray = None,
        q: np.ndarray = None,
        lmda: float = 0,
    ):
        """
        Initialize the latent factor predictor with given training data.

        Args:
            training_data (np.ndarray): The training data.
            k (int): The number of latent factors.
            p (np.ndarray, optional): The user latent factors.
            q (np.ndarray, optional): The item latent factors.
        """
        self.training_data = training_data
        self.k = k
        u, i = training_data.shape
        self.p = p if p is not None else np.random.rand(k, u) * 10.0 - 5
        self.q = q if q is not None else np.random.rand(k, i) * 10.0 - 5
        self.lmda = lmda
        self.train()

    def train(self, iterations: int = 20):
        for it in range(iterations):
            for i in range(self.training_data.shape[1]):
                u_i = list(
                    filter(
                        partial(
                            lambda u, i: not np.ma.is_masked(self.training_data[u, i]),
                            i=i,
                        ),
                        list(range(self.training_data.shape[0])),
                    )
                )
                left_sum = np.sum(
                    self.p[:, [u]] @ self.p[:, [u]].T for u in u_i
                ) + self.lmda * np.identity(self.k)
                right_sum = np.sum(
                    self.training_data[u, i] * self.p[:, [u]] for u in u_i
                )
                self.q[:, [i]] = np.linalg.solve(left_sum, right_sum)
            for u in range(self.training_data.shape[0]):
                i_u = list(
                    filter(
                        partial(
                            lambda i, u: not np.ma.is_masked(self.training_data[u, i]),
                            u=u,
                        ),
                        list(range(self.training_data.shape[1])),
                    )
                )
                left_sum = np.sum(
                    self.q[:, [i]] @ self.q[:, [i]].T for i in i_u
                ) + self.lmda * np.identity(self.k)
                right_sum = np.sum(
                    self.training_data[u, i] * self.q[:, [i]] for i in i_u
                )
                self.p[:, [u]] = np.linalg.solve(left_sum, right_sum)

    @override
    def predict(self, entries):
        res = np.zeros(len(entries))
        for idx, entry in enumerate(entries):
            u, i = entry
            res[idx] = self.p[:, [u]].T @ self.q[:, [i]]
        return res

    @override
    def predict_all(self):
        if self.prediction is not None:
            return self.prediction
        self.prediction = np.zeros(shape=self.training_data.shape, dtype=np.float64)
        for u in range(self.training_data.shape[0]):
            for i in range(self.training_data.shape[1]):
                self.prediction[u, i] = self.predict([(u, i)])
        return self.prediction
