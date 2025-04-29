from typing import override
import numpy as np
from tqdm import tqdm
from predictors import Predictor
from scipy.sparse import csr_matrix


class LatentFactorPredictor(Predictor):
    """
    A latent factor predictor trained with given training data.
    """

    k: int
    p: np.ndarray
    q: np.ndarray
    shape: tuple[int]
    lmda: float

    def __init__(
        self,
        k: int,
        shape: tuple[int],
        p: np.ndarray = None,
        q: np.ndarray = None,
        lmda: float = 0,
    ):
        """
        Initialize the latent factor predictor with given training data.

        Args:
            training_data (csr_matrix): The training data as a sparse matrix.
            k (int): The number of latent factors.
            p (np.ndarray, optional): The user latent factors.
            q (np.ndarray, optional): The item latent factors.
        """
        self.k = k
        self.shape = shape
        n, m = shape
        self.p = p if p is not None else np.random.rand(k, n) - 0.5
        self.q = q if q is not None else np.random.rand(k, m) - 0.5
        self.lmda = lmda

    @override
    def predict(self, entries, quiet=False):
        if not quiet:
            print("Predicting entries...")
        res = np.zeros(len(entries), dtype=np.float64)
        for idx, (u, i) in enumerate(tqdm(entries, disable=quiet)):
            # Dot product between user and item latent factors.
            res[idx] = self.p[:, [u]].T @ self.q[:, [i]]
        if not quiet:
            print("Finished predicting entries.")
        return np.clip(res, 0.5, 5)

    @override
    def predict_all(self, quiet=False):
        if not quiet:
            print("Predicting all...")
        n, m = self.shape
        prediction = np.zeros((n, m), dtype=np.float64)
        for u in tqdm(range(n), disable=quiet):
            for i in range(m):
                # For each (u, i) predict the rating.
                prediction[u, i] = self.predict([(u, i)], quiet=True)
        if not quiet:
            print("Finished predicting all.")
        return np.clip(prediction, 0.5, 5)

    @override
    def train(self, training_data: csr_matrix, max_iterations: int = 10000, tol=1e-5):
        """
        Optimized alternating least squares using sparse training data.

        Optimized by o3-mini from `old_train`.
        """
        print("Preparing data for training...")
        R = training_data
        n, m = self.shape
        if self.shape != R.shape:
            raise ValueError(
                f"Training data shape {R.shape} does not match predictor shape {self.shape}."
            )
        k = self.k
        lmda = self.lmda

        # Cache identity matrix for regularization
        I_k = np.eye(k, dtype=np.float64)

        print("Performing alternating least squares...")
        # ALS loop
        # Convert R to csc format for efficient column slicing during Q updates.
        R_csc = R.tocsc()

        for _ in tqdm(range(max_iterations)):
            prev_q = self.q.copy()
            prev_p = self.p.copy()
            # Update Q (item factors) using csc matrix for fast column access.
            for i in range(m):
                start, end = R_csc.indptr[i], R_csc.indptr[i + 1]
                users = R_csc.indices[start:end]
                if users.size == 0:
                    continue
                P_u = self.p[:, users]
                A = P_u @ P_u.T + lmda * I_k
                # Use the pre-stored nonzero ratings.
                r_vec = R_csc.data[start:end]
                b = P_u @ r_vec
                self.q[:, i] = np.linalg.solve(A, b)

            # Update P (user factors) using the fact that csr format excels at row slicing.
            for u in range(n):
                start, end = R.indptr[u], R.indptr[u + 1]
                items = R.indices[start:end]
                if items.size == 0:
                    continue
                Q_i = self.q[:, items]
                A = Q_i @ Q_i.T + lmda * I_k
                r_vec = R.data[start:end]
                b = Q_i @ r_vec
                self.p[:, u] = np.linalg.solve(A, b)
            # Check convergence
            if (
                np.linalg.norm(self.p - prev_p) < tol
                and np.linalg.norm(self.q - prev_q) < tol
            ):
                break
        print("Finished training.")

    def old_train(self, training_data: csr_matrix, iterations: int = 20):
        if self.shape != training_data.shape:
            raise ValueError(
                f"Training data shape {training_data.shape} does not match predictor shape {self.shape}."
            )
        for _ in tqdm(range(iterations)):
            for i in tqdm(range(self.shape[1]), leave=False):
                # Get users who rated item i.
                u_i = list(training_data.getcol(i).indices)
                if len(u_i) == 0:
                    continue
                left_sum = sum(
                    self.p[:, [u]] @ self.p[:, [u]].T for u in u_i
                ) + self.lmda * np.identity(self.k)
                right_sum = sum(
                    float(training_data[u, i]) * self.p[:, [u]] for u in u_i
                )
                self.q[:, i] = np.linalg.solve(left_sum, right_sum)
            for u in tqdm(range(self.shape[0]), leave=False):
                # Get items rated by user u.
                i_u = list(training_data.getrow(u).indices)
                if len(i_u) == 0:
                    continue
                left_sum = sum(
                    self.q[:, [i]] @ self.q[:, [i]].T for i in i_u
                ) + self.lmda * np.identity(self.k)
                right_sum = sum(
                    float(training_data[u, i]) * self.q[:, [i]] for i in i_u
                )
                self.p[:, u] = np.linalg.solve(left_sum, right_sum)
