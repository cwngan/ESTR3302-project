from enum import Enum
from typing import Any, override
import numpy as np
from predictors import Predictor
from predictors.least_squares import LeastSquaresPredictor
from utils.neighbor_selection import most_similar


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

    def __init__(
        self,
        correlation: Correlation,
        training_data: np.ndarray = None,
        lmda: float = None,
        baseline: LeastSquaresPredictor = None,
    ):
        if baseline is None:
            self.training_data = training_data
            self.lmda = lmda
            self.baseline = LeastSquaresPredictor(training_data, lmda)
        else:
            self.training_data = baseline.training_data
            self.lmda = baseline.lmda
            self.baseline = baseline
        self.correlation = correlation

    def _find_cosine_coefficients(self, data: np.ma):
        """
        Vectorized cosine–similarity (zeroing any position masked in either column).

        Optimized by o3-mini from `_old_find_cosin_coefficients`.
        """
        # if you want user–user rather than item–item, just transpose once
        if self.correlation == Correlation.USER:
            data = data.T

        # data may be a MaskedArray; extract mask and fill masked slots with 0
        mask = np.ma.getmask(data)  # True where missing
        X = data.filled(0.0)  # numeric array, zeros at masked

        # M[k,i] = 1.0 if data[k,i] is present, else 0.0
        M = (~mask).astype(np.float64)

        # 1) Numerator matrix: for each (i,j), sum_k X[k,i]*X[k,j] * (zeros handle masking)
        D_num = X.T @ X  # shape (m,m)

        # 2) We need for each (i,j) the norms of column i & j after zeroing any position masked in the other:
        #    S_i_j = sum_k X[k,i]^2 * M[k,j]
        #    S_j_i = sum_k X[k,j]^2 * M[k,i]
        sq = X * X
        S_i_j = sq.T @ M  # shape (m,m)
        # S_j_i = (M.T @ sq)  which is just S_i_j.T

        # 3) denominator matrix and final cosine
        denom = np.sqrt(S_i_j * S_i_j.T)  # elementwise sqrt
        # avoid divide-by-zero
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
    def predict(self, entries):
        if (
            self.neighbor_table is None
            or self.error is None
            or self.cosine_coefficients is None
        ):
            raise RuntimeError("Predictor has not been trained yet")
        result = self.baseline.predict(entries)
        error = self.error
        if self.correlation == Correlation.USER:
            error = error.T
        for idx, entry in enumerate(entries):
            u, i = entry
            if self.correlation == Correlation.USER:
                u, i = i, u
            d_sum = sum(
                abs(self.cosine_coefficients[i][j]) for j in self.neighbor_table[u][i]
            )
            for j in self.neighbor_table[u][i]:
                result[idx] += self.cosine_coefficients[i][j] / d_sum * error[u][j]
        return np.clip(result, min=1, max=5)

    @override
    def predict_all(self):
        if (
            self.neighbor_table is None
            or self.error is None
            or self.cosine_coefficients is None
        ):
            raise RuntimeError("Predictor has not been trained yet")
        n, m = self.training_data.shape
        error = self.error
        if self.correlation == Correlation.USER:
            n, m = m, n
            error = error.T
        result = (
            self.baseline.predict_all().copy()
            if self.correlation == Correlation.ITEM
            else self.baseline.predict_all().T.copy()
        )
        for u in range(n):
            for i in range(m):
                d_sum = sum(
                    abs(self.cosine_coefficients[i][j])
                    for j in self.neighbor_table[u][i]
                )
                for j in self.neighbor_table[u][i]:
                    result[u][i] += self.cosine_coefficients[i][j] / d_sum * error[u][j]
        predictions = np.clip(result, min=1, max=5)
        if self.correlation == Correlation.USER:
            predictions = predictions.T
        return predictions

    @override
    def train(self, get_neighbors: Any = most_similar):
        self.error = (
            np.ma.masked_less(self.training_data, 0) - self.baseline.predict_all()
        )
        self.cosine_coefficients = self._find_cosine_coefficients(self.error)
        self.neighbor_table = get_neighbors(
            training_data=(
                self.training_data
                if self.correlation == Correlation.ITEM
                else self.training_data.T
            ),
            cosine_coefficients=self.cosine_coefficients,
        )
