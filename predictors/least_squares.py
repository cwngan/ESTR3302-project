from typing import override
import numpy as np
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from tqdm import tqdm
from predictors import Predictor
from utils import average


class LeastSquaresPredictor(Predictor):
    """
    A least squares predictor trained with given training data.
    """

    lmda: float
    b: np.ndarray = None
    average_rating: float

    def __init__(self, training_data: np.ndarray, lmda: float):
        self.training_data = training_data
        self.lmda = lmda
        self.average_rating = average(self.training_data)

    @override
    def predict(self, entries):
        if self.b is None:
            raise RuntimeError("Predictor not trained yet.")
        output = [0.0] * len(entries)
        print("Predicting entries...")
        for idx, entry in enumerate(tqdm(entries)):
            i, j = entry
            output[idx] = (
                self.average_rating + self.b[i] + self.b[len(self.training_data) + j]
            )
        print("Finish predicting entries.")
        return np.clip(output, 1, 5)

    @override
    def predict_all(self):
        if self.b is None:
            raise RuntimeError("Predictor not trained yet.")
        print("Predicting all...")
        prediction = np.zeros(shape=self.training_data.shape, dtype=np.float64)
        for i in tqdm(range(np.size(self.training_data, 0))):
            for j in range(np.size(self.training_data, 1)):
                prediction[i][j] = (
                    self.average_rating
                    + self.b[i]
                    + self.b[len(self.training_data) + j]
                )
        print("Finish predicting all.")
        return prediction.clip(min=1, max=5)

    @override
    def train(self):
        """
        Train the predictor using the training data using sparse matrices.

        Optimized by o3-mini from `old_train`.
        """
        print("Constructing relevant matrices...", flush=True)
        # number of ratings per user u and per item i
        N_u = np.sum(~np.ma.getmask(self.training_data), axis=1)  # shape (n,)
        N_i = np.sum(~np.ma.getmask(self.training_data), axis=0)  # shape (m,)

        C = self.training_data - self.average_rating
        uc = np.array(C.sum(axis=1))
        ic = np.array(C.sum(axis=0))
        rhs = np.concatenate([uc, ic])

        U = diags(N_u + self.lmda)
        I = diags(N_i + self.lmda)
        # off‐diagonals: rating adjacency as sparse

        mask = ~np.ma.getmask(self.training_data)
        # build sparse mask matrix M of shape (n,m)
        M = csr_matrix(mask.astype(np.float64))
        AAT = bmat([[U, M], [M.T, I]])
        print("Calculating user and item biases...", flush=True)
        self.b = spsolve(AAT, rhs)
        print("Training done.")

    def old_train(self):
        """
        Train the predictor naively using the training data.
        """
        n, m = self.training_data.shape
        a = np.zeros(shape=(n * m, n + m), dtype=np.float64)
        c = np.zeros(shape=(n * m, 1), dtype=np.float64)

        curr_col = 0

        for i, row in enumerate(self.training_data):
            for j, col in enumerate(row):
                if np.ma.is_masked(col):
                    continue
                a[curr_col, i] = 1
                a[curr_col, n + j] = 1
                c[curr_col, 0] = col - self.average_rating
                curr_col += 1

        a = a[:curr_col, :]
        a_transpose = a.T
        c = c[:curr_col, :]
        self.b = np.linalg.solve(
            a_transpose @ a + self.lmda * np.identity(a_transpose.shape[0]),
            a_transpose @ c,
        )
