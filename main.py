"""Module for data predictions."""

import numpy as np

# sample_user_ratings = [
#     [5, 4, 5, 3, 3],
#     [3, 2, 2, 4, 1],
#     [3, 4, 3, 5, 4],
#     [5, 1, 4, 2, 4],
#     [2, 3, 4, 1, 1],
#     [2, 3, 4, 2, 5],
# ]

# sample_test_set = [
#     (0, 0),
#     (0, 3),
#     (1, 1),
#     (1, 4),
#     (2, 0),
#     (2, 4),
#     (3, 2),
#     (4, 1),
#     (4, 3),
#     (5, 0),
# ]
sample_user_ratings = [
    [5, 4, 4, None, 5],
    [None, 3, 5, 3, 4],
    [5, 2, None, 2, 3],
    [None, 2, 3, 1, 2],
    [4, None, 5, 4, 5],
    [5, 3, None, 3, 5],
    [3, 2, 3, 2, None],
    [5, 3, 4, None, 5],
    [4, 2, 5, 4, None],
    [5, None, 5, 3, 4],
]
sample_test_set = [
    (0, 4),
    (1, 3),
    (2, 3),
    (3, 1),
    (4, 2),
    (5, 0),
    (6, 1),
    (7, 1),
    (8, 0),
    (9, 0),
]
predictions = []


def average(data):
    """
    Find the average of a matrix ignoring NaN values.
    """
    masked_data = np.ma.masked_array(data, np.isnan(data))
    return np.average(masked_data)


def remove_test_set(training_set, test_set):
    """
    Returns a copy of the training set with items in test set set to NaN.
    """
    res = []
    for row in training_set:
        res.append([])
        for col in row:
            res[-1].append(col)
    for test in test_set:
        res[test[0]][test[1]] = None
    return np.array(res, dtype=float)


def get_test_set_matrix(training_set, test_set):
    res = np.full(
        (
            len(training_set),
            len(training_set[0]),
        ),
        np.nan,
        dtype=float,
    )
    for test in test_set:
        res[test[0]][test[1]] = training_set[test[0]][test[1]]
    return res


def find_MSE(prediction: np.ndarray[any], known: np.ndarray[2]):
    mse = 0
    ts = 0
    for i in range(len(prediction)):
        for j in range(len(prediction[i])):
            if np.isnan(prediction[i][j]) or np.isnan(known[i][j]):
                continue
            mse += (prediction[i][j] - known[i][j]) ** 2
            ts += 1
    return mse / ts


def find_baseline_prediction(training_set, lmda=0):
    """
    Perform baseline prediction with given training set.
    """
    average_rating = average(training_set)

    A = []
    c = []

    for i, row in enumerate(training_set):
        for j, col in enumerate(row):
            if np.isnan(col):
                continue
            tmp = [0] * (len(training_set) + len(row))
            tmp[i] = 1
            tmp[len(training_set) + j] = 1
            A.append(tmp)
            c.append(col - average_rating)

    A = np.array(A)
    c = np.atleast_2d(c).transpose()
    AT = A.transpose()
    b = np.linalg.solve(AT @ A + lmda * np.identity(AT.shape[0]), AT @ c)
    output = np.zeros(shape=(len(training_set), len(training_set[0])))
    for i in range(len(training_set)):
        for j in range(len(training_set[0])):
            output[i][j] = average_rating + b[i][0] + b[len(training_set) + j][0]
    np.clip(output, 1, 5, out=output)
    return output


def find_cosine_coefficients(data):
    """
    Calculate the cosine coefficient of every pair of columns in data.
    """
    print(data)
    m = len(data[0])
    D = np.empty(
        (
            m,
            m,
        )
    )
    for i in range(m):
        for j in range(m):
            if i == j:
                D[i][j] = 0
                continue
            ri = data[:, [i]]
            rj = data[:, [j]]
            ri[np.isnan(rj)] = 0
            rj[np.isnan(ri)] = 0
            ri[np.isnan(ri)] = 0
            rj[np.isnan(rj)] = 0
            D[i][j] = (ri.transpose() @ rj).sum()
            D[i][j] /= np.linalg.norm(ri) * np.linalg.norm(rj)
    return D


def find_improved_prediction(training_set, baseline_prediction, D, error_matrix):
    """
    Calculate the improved prediction of based on the baseline prediction,
    cosine coefficients, and error matrix.
    """
    n = len(training_set)
    m = len(training_set[0])
    # D_neighbors = np.argpartition(np.abs(D), m - 2)[:, m - 2:]
    D_neighbors = np.argsort(np.abs(D), axis=1)
    # D_neighbors = np.argpartition(np.abs(D), m - 1)[:, m - 1:]
    L = np.zeros(
        (
            n,
            m,
        )
    )
    for u in range(n):
        for i in range(m):
            d = D_neighbors[i]
            d_sum = 0
            k = m - 1
            while np.isnan(training_set[u][d[k]]):
                k -= 1
            d_sum += np.abs(D[i][d[k]])
            if d_sum == 0:
                continue
            L[u][i] += (D[i][d[k]] / d_sum) * error_matrix[u][d[k]]
    return np.clip(baseline_prediction + L, 1, 5)


if __name__ == "__main__":
    clean_training_set = remove_test_set(sample_user_ratings, sample_test_set)
    clean_test_set_matrix = get_test_set_matrix(sample_user_ratings, sample_test_set)
    baseline = find_baseline_prediction(clean_training_set, lmda=0.25)
    print(baseline)
    # print(remove_test_set(baseline, sample_test_set))
    print(find_MSE(baseline, clean_training_set) ** 0.5)
    print(clean_test_set_matrix)
    print(find_MSE(baseline, clean_test_set_matrix) ** 0.5)
    error = clean_training_set - baseline
    print(error)
    cosine_coefficients = find_cosine_coefficients(error.transpose())
    np.set_printoptions(formatter={"float": lambda x: "{0:0.4f}".format(x)})
    print(cosine_coefficients)
    improved = find_improved_prediction(
        clean_training_set.transpose(),
        baseline.transpose(),
        cosine_coefficients,
        error.transpose(),
    )
    print(improved.transpose())
    # print(find_MSE(improved, clean_training_set) ** 0.5)
    # print(clean_test_set_matrix)
    # print(find_MSE(improved, clean_test_set_matrix) ** 0.5)
