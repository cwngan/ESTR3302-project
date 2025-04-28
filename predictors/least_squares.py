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

    def __init__(self, training_data: csr_matrix, lmda: float):
        self.training_data = training_data
        self.lmda = lmda
        self.average_rating = average(self.training_data)

    @override
    def predict(self, entries, quiet=False):
        if self.b is None:
            raise RuntimeError("Predictor not trained yet.")
        output = [0.0] * len(entries)
        if not quiet:
            print("Predicting entries...")
        for idx, entry in enumerate(tqdm(entries, disable=quiet)):
            i, j = entry
            output[idx] = (
                self.average_rating
                + self.b[i]
                + self.b[self.training_data.shape[0] + j]
            )
        if not quiet:
            print("Finished predicting entries.")
        return np.clip(output, 1, 5)

    @override
    def predict_all(self, quiet=False):
        if self.b is None:
            raise RuntimeError("Predictor not trained yet.")
        if not quiet:
            print("Predicting all...")
        rows, cols = self.training_data.nonzero()
        data = []
        for i, j in tqdm(zip(rows, cols), total=len(rows), disable=quiet):
            pred = (
                self.average_rating
                + self.b[i]
                + self.b[self.training_data.shape[0] + j]
            )
            data.append(pred)
        data = np.clip(data, 1, 5)
        prediction = csr_matrix((data, (rows, cols)), shape=self.training_data.shape)
        if not quiet:
            print("Finished predicting all.")
        return prediction

    @override
    def train(self):
        """
        Train the predictor using the training data using sparse matrices.

        Optimized by o3-mini from `old_train`.
        """
        print("Constructing relevant matrices...", flush=True)
        # number of ratings per user u and per item i (non-zero entries)
        N_u = self.training_data.getnnz(axis=1)  # shape (n,)
        N_i = self.training_data.getnnz(axis=0)  # shape (m,)

        C = self.training_data.copy()
        C.data = C.data - self.average_rating
        uc = np.array(C.sum(axis=1)).flatten()
        ic = np.array(C.sum(axis=0)).flatten()
        rhs = np.concatenate([uc, ic])

        U = diags(N_u + self.lmda)
        I = diags(N_i + self.lmda)
        # off‐diagonals: rating adjacency as sparse
        # build sparse mask matrix M where nonzero entries indicate known ratings
        M = self.training_data.copy()
        M.data = np.ones_like(M.data)
        AAT = bmat([[U, M], [M.T, I]], format="csr")
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
