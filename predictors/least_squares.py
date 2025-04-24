from typing import override
import numpy as np

from predictors import Predictor
from utils import average


class LeastSquaresPredictor(Predictor):
    """
    A least squares predictor trained with given training data.
    """

    lmda: float
    b: np.ndarray
    average_rating: float
    prediction: np.ndarray = None

    def __init__(self, training_data: np.ndarray, lmda: float):
        self.training_data = training_data
        self.lmda = lmda
        self.average_rating = average(self.training_data)
        self._find_bias()

    def _find_bias(self):
        """
        Perform baseline prediction with given training set.
        """
        a = []
        c = []

        for i, row in enumerate(self.training_data):
            for j, col in enumerate(row):
                if np.ma.is_masked(col):
                    continue
                tmp = [0] * (len(self.training_data) + len(row))
                tmp[i] = 1
                tmp[len(self.training_data) + j] = 1
                a.append(tmp)
                c.append(col - self.average_rating)

        a = np.array(a)
        c = np.atleast_2d(c).T
        a_transpose = a.T
        self.b = np.linalg.solve(
            a_transpose @ a + self.lmda * np.identity(a_transpose.shape[0]),
            a_transpose @ c,
        )

    @override
    def predict(self, entries):
        if self.prediction is None:
            raise RuntimeError("Predictor has not been trained yet")
        return [self.prediction[i][j] for i, j in entries]

    @override
    def predict_all(self):
        if self.prediction is None:
            raise RuntimeError("Predictor has not been trained yet")
        return self.prediction

    @override
    def train(self):
        prediction = np.zeros(shape=self.training_data.shape, dtype=np.float64)
        for i in range(np.size(self.training_data, 0)):
            for j in range(np.size(self.training_data, 1)):
                prediction[i][j] = (
                    self.average_rating
                    + self.b[i][0]
                    + self.b[len(self.training_data) + j][0]
                )
        self.prediction = prediction.clip(min=1, max=5)
