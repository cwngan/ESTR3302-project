from functools import partial
from typing import override
import numpy as np
from tqdm import tqdm
from predictors import Predictor


class LatentFactorPredictor(Predictor):
    """
    A latent factor predictor trained with given training data.
    """

    k: int
    p: np.ndarray
    q: np.ndarray
    lmda: float

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

    @override
    def predict(self, entries, quiet=False):
        if not quiet:
            print("Predicting entries...")
        res = np.zeros(len(entries))
        for idx, entry in enumerate(tqdm(entries, disable=quiet)):
            u, i = entry
            res[idx] = self.p[:, [u]].T @ self.q[:, [i]]
        if not quiet:
            print("Finished predicting entries.")
        return res

    @override
    def predict_all(self, quiet=False):
        if not quiet:
            print("Predicting all...")
        prediction = np.zeros(shape=self.training_data.shape, dtype=np.float64)
        for u in tqdm(range(self.training_data.shape[0]), disable=quiet):
            for i in range(self.training_data.shape[1]):
                prediction[u, i] = self.predict([(u, i)], quiet=True)
        if not quiet:
            print("Finished predicting all.")
        return prediction

    @override
    def train(self, iterations: int = 20):
        """
        Optimized by o3-mini from `old_train`.
        """
        print("Preparing data for training...")
        R = self.training_data
        n, m = R.shape
        k = self.k
        lmda = self.lmda

        # 1) precompute for each item i the list of users who rated it
        #    and for each user u the list of items they rated
        mask = np.ma.getmaskarray(R)
        users_by_item = [np.nonzero(~mask[:, i])[0] for i in range(m)]
        items_by_user = [np.nonzero(~mask[u, :])[0] for u in range(n)]

        # 2) cache I_k
        I_k = np.eye(k, dtype=np.float64)

        print("Performing alternating least squares...")
        # 3) ALS loop
        for _ in tqdm(range(iterations)):
            # update Q (item factors)
            for i, users in enumerate(users_by_item):
                if users.size == 0:
                    continue
                # P_u: k×|U_i|
                P_u = self.p[:, users]
                # A = P_u P_u^T + λ I
                A = P_u @ P_u.T + lmda * I_k
                # b = sum_{u∈U_i} r_{u,i} * P_u[:,u]
                #    = P_u @ r_vec  where r_vec has shape (|U_i|,)
                r_vec = R[users, i].astype(np.float64)
                b = P_u @ r_vec
                # solve A q_i = b
                self.q[:, i] = np.linalg.solve(A, b)

            # update P (user factors)
            for u, items in enumerate(items_by_user):
                if items.size == 0:
                    continue
                Q_i = self.q[:, items]
                A = Q_i @ Q_i.T + lmda * I_k
                r_vec = R[u, items].astype(np.float64)
                b = Q_i @ r_vec
                self.p[:, u] = np.linalg.solve(A, b)
        print("Finished training.")

    def old_train(self, iterations: int = 20):
        for _ in tqdm(range(iterations)):
            for i in tqdm(range(self.training_data.shape[1]), leave=False):
                u_i = list(
                    filter(
                        partial(
                            lambda u, i: not np.ma.is_masked(self.training_data[u, i]),
                            i=i,
                        ),
                        list(range(self.training_data.shape[0])),
                    )
                )
                if len(u_i) == 0:
                    continue
                left_sum = np.sum(
                    self.p[:, [u]] @ self.p[:, [u]].T for u in u_i
                ) + self.lmda * np.identity(self.k)
                right_sum = np.sum(
                    self.training_data[u, i] * self.p[:, [u]] for u in u_i
                )
                self.q[:, [i]] = np.linalg.solve(left_sum, right_sum)
            for u in tqdm(range(self.training_data.shape[0]), leave=False):
                i_u = list(
                    filter(
                        partial(
                            lambda i, u: not np.ma.is_masked(self.training_data[u, i]),
                            u=u,
                        ),
                        list(range(self.training_data.shape[1])),
                    )
                )
                if len(i_u) == 0:
                    continue
                left_sum = np.sum(
                    self.q[:, [i]] @ self.q[:, [i]].T for i in i_u
                ) + self.lmda * np.identity(self.k)
                right_sum = np.sum(
                    self.training_data[u, i] * self.q[:, [i]] for i in i_u
                )
                self.p[:, [u]] = np.linalg.solve(left_sum, right_sum)
