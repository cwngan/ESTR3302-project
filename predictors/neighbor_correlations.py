from enum import Enum
from typing import Any, override
import numpy as np
from tqdm import tqdm
from predictors import Predictor
from predictors.least_squares import LeastSquaresPredictor
from utils.neighbor_selection import most_similar
from scipy.sparse import csr_matrix


class Correlation(Enum):
    ITEM = 0
    USER = 1


class NeighborCorrelationsPredictor(Predictor):
    """
    A predictor that uses neighbor correlations to make predictions.
    Uses LeastSqauresPredictor as the baseline predictor as default.
    """

    lmda: float
    baseline: Predictor
    error: np.ndarray = None
    cosine_coefficients: np.ndarray = None
    correlation: Correlation
    neighbor_table: np.ndarray = None
    _training_data: csr_matrix = None

    def __init__(
        self,
        correlation: Correlation,
        shape: tuple[int] = None,
        lmda: float = None,
        baseline: LeastSquaresPredictor = None,
    ):
        """
        Initialize the neighbor correlations improved predictor.

        Args:
            correlation (Correlation): The type of correlation to use (item or user).
            shape (tuple[int], optional): The shape of the training data.
            lmda (float, optional): The regularization parameter.
            baseline (LeastSquaresPredictor, optional): The baseline predictor.
        """
        if baseline is None:
            self.lmda = lmda
            self.baseline = LeastSquaresPredictor(lmda, shape)
        else:
            self.lmda = baseline.lmda
            self.baseline = baseline
        self.shape = shape
        self.correlation = correlation

    def _find_cosine_coefficients(self, data: np.ma.masked_array):
        """
        Vectorized cosine–similarity (zeroing any position masked in either column).

        Optimized by o3-mini from `_old_find_cosin_coefficients`.
        """
        # if you want user–user rather than item–item, just transpose once
        if self.correlation == Correlation.USER:
            data = data.T

        X = data.filled(0.0)  # numeric array, zeros at masked

        # Numerator matrix: for each (i,j), sum_k X[k,i]*X[k,j] * (zeros handle masking)
        D_num = X.T @ X  # shape (m,m)

        # Extract mask and fill masked slots with 0
        mask = np.ma.getmask(data)  # True where missing
        # M[k,i] = 1.0 if data[k,i] is present, else 0.0
        M = (~mask).astype(np.float64)

        sq = X * X
        S_ij = sq.T @ M  # shape (m,m)

        # Denominator matrix and final cosine
        denom = np.sqrt(S_ij * S_ij.T)  # elementwise sqrt
        # Avoid divide-by-zero
        denom[denom == 0] = 1.0
        D = D_num / denom

        # zero out diagonal (self-similarity)
        np.fill_diagonal(D, 0.0)

        return D

    def _old_find_cosine_coefficients(self, data: np.ndarray):
        """
        Calculate the cosine coefficient of every pair of columns in data.
        """
        if self.correlation == Correlation.USER:
            data = data.T
        m = len(data[0])
        D = np.empty((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    D[i][j] = 0
                    continue
                ri = data[:, [i]]
                rj = data[:, [j]]
                ri[np.ma.getmask(rj)] = 0
                rj[np.ma.getmask(ri)] = 0
                ri[np.ma.getmask(ri)] = 0
                rj[np.ma.getmask(rj)] = 0
                D[i][j] = (ri.T @ rj).sum()
                D[i][j] /= np.linalg.norm(ri) * np.linalg.norm(rj)
        return D

    @override
    def predict(self, entries, quiet=False):
        if (
            self.neighbor_table is None
            or self.error is None
            or self.cosine_coefficients is None
        ):
            raise RuntimeError("Predictor has not been trained yet")
        print("Predicting entries...")
        result = self.baseline.predict(entries, quiet=True)
        error = self.error
        training_data = self._training_data
        if self.correlation == Correlation.USER:
            error = error.T
            training_data = training_data.T
        for idx, entry in enumerate(tqdm(entries, disable=quiet)):
            u, i = entry
            if self.correlation == Correlation.USER:
                u, i = i, u
            if not isinstance(self.neighbor_table[u][i], (int, np.int64)):
                d_sum = sum(
                    abs(self.cosine_coefficients[i][j])
                    for j in self.neighbor_table[u][i]
                )
                if d_sum == 0:
                    continue
                for j in self.neighbor_table[u][i]:
                    result[idx] += self.cosine_coefficients[i][j] / d_sum * error[u][j]
            else:
                j = self.neighbor_table[u][i]
                if np.ma.is_masked(training_data[u][j]):
                    continue
                d_sum = abs(self.cosine_coefficients[i][j])
                if d_sum == 0:
                    continue
                result[idx] += self.cosine_coefficients[i][j] / d_sum * error[u][j]
        print("Finished predicting entries.")
        return np.clip(result, min=0.5, max=5)

    @override
    def predict_all(self, quiet=False):
        if (
            self.neighbor_table is None
            or self.error is None
            or self.cosine_coefficients is None
        ):
            raise RuntimeError("Predictor has not been trained yet")
        print("Predicting all...")
        n, m = self.shape
        training_data = self._training_data
        error = self.error
        if self.correlation == Correlation.USER:
            n, m = m, n
            error = error.T
            training_data = training_data.T
        result = (
            self.baseline.predict_all(quiet=True).copy()
            if self.correlation == Correlation.ITEM
            else self.baseline.predict_all(quiet=True).T.copy()
        )
        for u in tqdm(range(n), disable=quiet):
            for i in range(m):
                if not isinstance(self.neighbor_table[u][i], int):
                    d_sum = sum(
                        abs(self.cosine_coefficients[i][j])
                        for j in self.neighbor_table[u][i]
                    )
                    if d_sum == 0:
                        continue
                    for j in self.neighbor_table[u][i]:
                        result[u][i] += (
                            self.cosine_coefficients[i][j] / d_sum * error[u][j]
                        )
                else:
                    j = self.neighbor_table[u][i]
                    if np.ma.is_masked(training_data[u][j]):
                        continue
                    d_sum = abs(self.cosine_coefficients[i][j])
                    if d_sum == 0:
                        continue
                    result[u][i] += self.cosine_coefficients[i][j] / d_sum * error[u][j]
        predictions = np.clip(result, min=0.5, max=5)
        if self.correlation == Correlation.USER:
            predictions = predictions.T
        print("Finished predicting all.")
        return predictions

    @override
    def train(self, training_data: csr_matrix, get_neighbors: Any = most_similar):
        print("Calculating cosine similarity coefficients...")
        self._training_data = np.ma.masked_equal(training_data.toarray(), 0)
        self.error = np.ma.masked_equal(
            self._training_data, 0
        ) - self.baseline.predict_all(quiet=True)
        self.cosine_coefficients = self._find_cosine_coefficients(self.error)
        print("Making neighbor table...")
        self.neighbor_table = get_neighbors(
            training_data=(
                self._training_data
                if self.correlation == Correlation.ITEM
                else self._training_data.T
            ),
            cosine_coefficients=self.cosine_coefficients,
        )
        print("Finished training.")
