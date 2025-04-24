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

    def _find_cosine_coefficients(self, data: np.ndarray):
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
