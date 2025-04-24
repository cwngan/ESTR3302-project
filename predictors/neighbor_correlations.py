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
    error: np.ndarray
    cosine_coefficients: np.ndarray
    correlation: Correlation
    predictions: np.ndarray = None

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
        self.error = (
            np.ma.masked_less(self.training_data, 0) - self.baseline.predict_all()
        )
        self.correlation = correlation
        self.cosine_coefficients = (
            self._find_cosine_coefficients(self.error)
            if self.correlation == Correlation.ITEM
            else self._find_cosine_coefficients(self.error.T)
        )

    def _find_cosine_coefficients(self, data: np.ndarray):
        """
        Calculate the cosine coefficient of every pair of columns in data.
        """
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
    def predict(self, entries, get_neighbors: Any = most_similar):
        if self.predictions is not None:
            return self.predictions[entries[0]][entries[1]]
        result = self.baseline.predict(entries)
        for idx, entry in enumerate(entries):
            neighbors = get_neighbors(
                training_data=self.training_data,
                cosine_coefficients=self.cosine_coefficients,
                one=True,
                entry=entry,
            )
            u, i = entry
            d_sum = sum(abs(self.cosine_coefficients[i][j]) for j in neighbors)
            for j in neighbors:
                result[idx] += self.cosine_coefficients[i][j] / d_sum * self.error[u][j]
        return np.clip(result, min=1, max=5)

    @override
    def predict_all(self, get_neighbors: Any = most_similar):
        """
        Calculate the improved prediction of based on the baseline prediction,
        cosine coefficients, and error matrix.
        """
        if self.predictions is not None:
            return self.predictions
        n, m = self.training_data.shape
        L = get_neighbors(
            training_data=self.training_data,
            cosine_coefficients=self.cosine_coefficients,
        )
        result = self.baseline.predict_all().copy()
        for u in range(n):
            for i in range(m):
                d_sum = sum(abs(self.cosine_coefficients[i][j]) for j in L[u][i])
                for j in L[u][i]:
                    result[u][i] += (
                        self.cosine_coefficients[i][j] / d_sum * self.error[u][j]
                    )
        self.predictions = np.clip(result, min=1, max=5)
        return self.predictions
