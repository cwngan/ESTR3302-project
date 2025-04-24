from typing import Any
import numpy as np


def mask_unknown_values(training_data: Any):
    res = np.array(training_data, dtype=np.float64)
    mask = np.less(training_data, 0)
    return np.ma.masked_array(res, np.ma.make_mask(mask))


def get_test_set_mask(test_set: list[tuple[int]], shape: tuple[int]):
    """
    Returns a mask for the test set.
    """
    mask = np.zeros(shape=shape)
    for i, j in test_set:
        mask[i, j] = 1
    return np.ma.make_mask(mask)


def average(data: np.ndarray):
    """
    Find the average of a matrix ignoring negative values.
    """
    masked_data = np.ma.masked_less(data, 0)
    return np.average(masked_data)


def remove_test_set(training_data: Any, test_set: list[tuple[int]]):
    """
    Returns a copy of the training set with items in test set set to -1.
    """
    res = mask_unknown_values(training_data)
    return np.ma.masked_array(res, get_test_set_mask(test_set, res.shape))


def get_test_set_matrix(training_data: Any, test_set: list[tuple[int]]):
    """
    Returns a matrix representing the data of the test set.
    """
    res = np.array(training_data, dtype=np.float64)
    return np.ma.masked_array(res, ~get_test_set_mask(test_set, res.shape))


def mean_square_error(prediction: np.ndarray, known: np.ndarray):
    """
    Calculate the mean square error between prediction and known values.
    """
    mse = 0
    ts = 0
    for i, row in enumerate(prediction):
        for j, cell in enumerate(row):
            if np.ma.is_masked(cell) or np.ma.is_masked(known[i][j]):
                continue
            mse += (cell - known[i][j]) ** 2
            ts += 1
    return mse / ts


def root_mean_square_error(prediction: np.ndarray, known: np.ndarray):
    """
    Calculate the root mean square error between prediction and known values.
    """
    return mean_square_error(prediction, known) ** 0.5
