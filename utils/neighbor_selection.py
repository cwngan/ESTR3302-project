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
        while np.ma.is_masked(training_data[u][d[k]]):
            k -= 1
        return [d[k]]
    L = [[[] for _ in range(m)] for __ in range(n)]
    for u in tqdm(range(n)):
        for i in range(m):
            d = D_neighbors[i]
            k = m - 1
            while np.ma.is_masked(training_data[u][d[k]]):
                k -= 1
            L[u][i].append(d[k])
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
        for i in range(m):
            d = D_neighbors[i]
            k = m - 1
            while k >= 0 and len(L[u][i]) < 2:
                if not np.ma.is_masked(training_data[u][d[k]]):
                    L[u][i].append(d[k])
                k -= 1
    return L


def two_most_similar_skip_masked(
    training_data: np.ndarray,
    cosine_coefficients: np.ndarray,
    one: bool = False,
    entry: tuple[int] = None,
):
    """
    Return a list representing the two most similar entry of each entry
    in `training_data` according to their cosine coefficients.

    Entries that are masked in `training_data` are skipped, which means
    some entry may only has 1 or 0 neighbor.
    """
    n, m = training_data.shape
    D_neighbors = np.argsort(np.abs(cosine_coefficients), axis=1)
    if one:
        u, i = entry
        res = []
        d = D_neighbors[i]
        k = m - 1
        while k >= m - 2:
            if not np.ma.is_masked(training_data[u][d[k]]):
                res.append(d[k])
            k -= 1
        return res
    L = [[[] for _ in range(m)] for __ in range(n)]
    for u in tqdm(range(n)):
        for i in range(m):
            d = D_neighbors[i]
            k = m - 1
            while k >= m - 2:
                if not np.ma.is_masked(training_data[u][d[k]]):
                    L[u][i].append(d[k])
                k -= 1
    return L
