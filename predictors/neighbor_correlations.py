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

    def __init__(
        self,
        correlation: Correlation,
        training_data: csr_matrix = None,
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

    def _find_cosine_coefficients(self, data: csr_matrix):
        """
        Vectorized cosine–similarity (ignoring values equal to 0 as missing).
        Uses sparse matrix multiplication to optimize computation.

        Optimized by o3-mini from `_old_find_cosin_coefficients`.
        """
        # If user–user similarity is needed, work on transposed data
        if self.correlation == Correlation.USER:
            data = data.T

        # The numerator: dot product of columns (only nonzero entries contribute)
        numerator = data.T @ data  # sparse matrix multiplication
        numerator = numerator.toarray()  # convert to dense array

        # Compute the Euclidean norm for each column (only nonzero entries count)
        squared = data.copy()
        squared.data **= 2
        norm = np.sqrt(np.array(squared.sum(axis=0)).ravel())

        # Outer product of norms for every pair of columns
        denom = np.outer(norm, norm)
        denom[denom == 0] = 1.0  # avoid division by zero

        # Final cosine similarity matrix
        cosine = numerator / denom

        # Zero out the diagonal (self-similarity)
        np.fill_diagonal(cosine, 0.0)

        return cosine

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
        training_data = self.training_data
        if self.correlation == Correlation.USER:
            error = error.T
            training_data = training_data.T
        for idx, entry in enumerate(tqdm(entries, disable=quiet)):
            u, i = entry
            if self.correlation == Correlation.USER:
                u, i = i, u
            if not isinstance(self.neighbor_table[u][i], int):
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
        n, m = self.training_data.shape
        training_data = self.training_data
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
    def train(self, get_neighbors: Any = most_similar):
        print("Calculating cosine similarity coefficients...")
        self.error = np.ma.masked_less(
            self.training_data, 0
        ) - self.baseline.predict_all(quiet=True)
        self.cosine_coefficients = self._find_cosine_coefficients(self.error)
        print("Making neighbor table...")
        self.neighbor_table = get_neighbors(
            training_data=(
                self.training_data
                if self.correlation == Correlation.ITEM
                else self.training_data.T
            ),
            cosine_coefficients=self.cosine_coefficients,
        )
        print("Finished training.")
