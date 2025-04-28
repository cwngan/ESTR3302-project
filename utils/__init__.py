from typing import Any
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from tqdm import tqdm


def get_test_set_mask(test_set: list[tuple[int]], shape: tuple[int]):
    """
    Returns a mask for the test set.
    """
    mask = np.zeros(shape=shape)
    for i, j in test_set:
        mask[i, j] = 1
    return np.ma.make_mask(mask)


def average(data: csr_matrix):
    """
    Find the average of a csr_matrix ignoring stored zero and negative values.
    Only nonzero positive values are averaged.
    """
    # Extract the non-zero stored entries from the matrix
    if not isinstance(data, csr_matrix):
        raise ValueError("Input must be a csr_matrix")

    # Filter out zero and negative values from the stored data
    positive_entries = data.data[data.data > 0]
    if positive_entries.size == 0:
        return 0
    return np.average(positive_entries)


def remove_test_set(training_data: Any, test_set: list[tuple[int]]):
    """
    Returns a copy of the training set (a csr matrix) with items in the test set set to 0.
    This version groups test_set indices by row to reduce Python looping.
    """

    print("Removing test set entries from training data...")
    # Work directly with the CSR matrix
    res = training_data.copy()

    # Group test_set indices by row
    rows_to_cols = defaultdict(list)
    for i, j in tqdm(test_set):
        rows_to_cols[i].append(j)

    # For each affected row, set matching entries to 0
    for i, cols in tqdm(rows_to_cols.items()):
        start, end = res.indptr[i], res.indptr[i + 1]
        row_cols = res.indices[start:end]
        cols_array = np.array(cols, dtype=row_cols.dtype)
        mask = np.isin(row_cols, cols_array)
        res.data[start:end][mask] = 0

    res.eliminate_zeros()
    print("Done removing test set entries.")
    return res.tocsr()


def get_test_set_matrix(training_data: Any, test_set: list[tuple[int]]):
    """
    Returns a csr matrix representing the data of the test set.
    Only the entries specified in test_set are retained.
    """

    print("Getting test set matrix...")
    # Group test_set indices by row for fast lookup.
    rows_to_cols = defaultdict(set)
    for i, j in tqdm(test_set):
        rows_to_cols[i].add(j)

    # Build the data for the new sparse matrix.
    row_indices = []
    col_indices = []
    values = []

    # Iterate only over rows present in the test_set.
    for i, cols_set in tqdm(rows_to_cols.items()):
        start, end = training_data.indptr[i], training_data.indptr[i + 1]
        row_cols = training_data.indices[start:end]
        row_data = training_data.data[start:end]
        mask = np.isin(row_cols, list(cols_set))
        if np.any(mask):
            row_indices.extend([i] * np.count_nonzero(mask))
            col_indices.extend(row_cols[mask])
            values.extend(row_data[mask])

    print("Done getting test set matrix.")
    return csr_matrix((values, (row_indices, col_indices)), shape=training_data.shape)


def csr_get_entries(matrix: csr_matrix, entries: list[tuple[int]]) -> list[float]:
    """
    Helper function to extract values from a csr_matrix at given (row, col) entries.
    """
    result = []
    for r, c in entries:
        row_start = matrix.indptr[r]
        row_end = matrix.indptr[r + 1]
        # Use binary search on the sorted indices for this row.
        pos = np.searchsorted(matrix.indices[row_start:row_end], c)
        if pos < (row_end - row_start) and matrix.indices[row_start + pos] == c:
            result.append(matrix.data[row_start + pos])
        else:
            result.append(0)
    return result


def mean_square_error(prediction: csr_matrix, known: csr_matrix):
    """
    Calculate the mean square error between prediction and known values.
    All entries in the matrices are compared. The error is computed as the
    sum over squared differences divided by the total number of elements.
    """
    diff = prediction - known
    diff_array = np.array(diff.data)
    # Compute element-wise square of the differences.
    mse = (diff_array * diff_array).sum()
    return mse / len(diff_array)


def mean_square_error_entries(
    entry_prediction: np.ndarray, entries: list[tuple[int]], known: csr_matrix
):
    """
    Calculate the mean square error between entry_prediction and known values
    at the specified (row, col) entries.
    """
    # Retrieve values for each requested entry.
    pred_vals = entry_prediction
    known_vals = np.array(csr_get_entries(known, entries))
    mse = ((pred_vals - known_vals) ** 2).sum()
    return mse / len(entries)


def root_mean_square_error(prediction: csr_matrix, known: csr_matrix):
    """
    Calculate the root mean square error between prediction and known values.
    """
    return mean_square_error(prediction, known) ** 0.5


def root_mean_square_error_entries(
    entry_prediction: csr_matrix, entries: list[tuple[int]], known: csr_matrix
):
    """
    Calculate the root mean square error between entry_prediction and known values
    at the specified (row, col) entries.
    """
    return mean_square_error_entries(entry_prediction, entries, known) ** 0.5
