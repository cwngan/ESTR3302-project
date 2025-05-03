import numpy as np
from tqdm import tqdm


def most_similar(
    training_data: np.ndarray,
    cosine_coefficients: np.ndarray,
    one: bool = False,
    entry: tuple[int] = None,
):
    """
    Return a list representing the most similar entry of each entry
    in `training_data` according to their cosine coefficients.
    """
    n, m = training_data.shape
    D_neighbors = np.argsort(np.abs(cosine_coefficients), axis=1)
    if one:
        u, i = entry
        d = D_neighbors[i]
        k = m - 1
        while k >= 0 and np.ma.is_masked(training_data[u][d[k]]):
            k -= 1
        if k < 0:
            return []
        return [d[k]]
    L = np.zeros((n, m), dtype=int)
    for u in tqdm(range(n)):
        masked_d = np.ma.masked_array(
            np.abs(cosine_coefficients),
            np.tile(np.ma.getmaskarray(training_data[u]), (m, 1)),
        )
        L[u, :] = np.ma.argmax(masked_d, axis=1, fill_value=-999999)
    return L


def two_most_similar(
    training_data: np.ndarray,
    cosine_coefficients: np.ndarray,
    one: bool = False,
    entry: tuple[int] = None,
):
    """
    Return a list representing the two most similar entry of each entry
    in `training_data` according to their cosine coefficients.
    """
    n, m = training_data.shape
    D_neighbors = np.argsort(np.abs(cosine_coefficients), axis=1)
    if one:
        u, i = entry
        res = []
        d = D_neighbors[i]
        k = m - 1
        while k >= 0 and len(res) < 2:
            if not np.ma.is_masked(training_data[u][d[k]]):
                res.append(d[k])
            k -= 1
        return res
    L = [[[] for _ in range(m)] for __ in range(n)]
    for u in tqdm(range(n)):
        masked_d = np.ma.masked_array(
            np.abs(cosine_coefficients),
            np.tile(np.ma.getmaskarray(training_data[u]), (m, 1)),
        )
        D_neighbors = np.argpartition(
            np.ma.filled(np.ma.abs(masked_d), -999999), m - 2, axis=1
        )
        L[u] = D_neighbors[:, -2:]
    return L


def two_most_similar_skip_masked(
    training_data: csr_matrix,
    cosine_coefficients: np.ndarray,
    one: bool = False,
    entry: tuple[int, int] | None = None,
):
    """
    Return a list representing the two most similar entry for each entry (or for a given one)
    according to their cosine coefficients.

    For sparse training_data (scipy.sparse csr_matrix), an entry is considered "non-masked"
    if it exists in the sparse structure (i.e. if its column is in the row's indices).
    Note: This assumes that a missing entry (i.e. not stored) corresponds to a masked entry.
    """
    n, m = training_data.shape
    D_neighbors = np.argsort(np.abs(cosine_coefficients), axis=1)
    if one:
        if entry is None:
            raise ValueError("entry must be provided when one is True")
        u, i = entry
        res = []
        d = D_neighbors[i]
        # Check only the top two candidates (largest cosine coefficients)
        k = m - 1
        while k >= m - 2 and len(res) < 2:
            # In a csr_matrix, training_data[u].indices contains the column indices of nonzero entries.
            if d[k] in training_data[u].indices:
                res.append(d[k])
            k -= 1
        return res
    # For every row u, for every column i in D_neighbors, store the two most similar neighbors
    L = [[[] for _ in range(m)] for __ in range(n)]
    for u in tqdm(range(n)):
        # Precompute the set of non-masked (i.e. stored) column indices for this row.
        non_masked = set(training_data[u].indices)
        for i in range(m):
            d = D_neighbors[i]
            k = m - 1
            count = 0
            while k >= m - 2 and count < 2:
                if d[k] in non_masked:
                    L[u][i].append(d[k])
                    count += 1
                k -= 1
    return L
