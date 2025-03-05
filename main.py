"""Module for data predictions."""
import numpy as np

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
    [5, None, 5, 3, 4]
]

sample_test_set = [(0, 4), (1, 3), (2, 3), (3, 1), (4, 2),
                   (5, 0), (6, 1), (7, 1), (8, 0), (9, 0)]
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


def baseline_prediction(training_set, lmda=0):
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
            output[i][j] = (average_rating + b[i][0] +
                            b[len(training_set) + j][0])
    np.clip(output, 1, 5, out=output)
    return output


def cosine_coefficient(data):
    """
    Calculate the cosine coefficient of every pair of columns in data.
    """
    m = len(data[0])
    D = np.empty((m, m,))
    e2 = np.copy(data)
    for i in range(m):
        for j in range(m):
            # if i == j:
            #     D[i][j] = np.nan
            #     continue
            ri = e2[:, [i]]
            rj = e2[:, [j]]
            ri[np.isnan(rj)] = 0
            rj[np.isnan(ri)] = 0
            ri[np.isnan(ri)] = 0
            rj[np.isnan(rj)] = 0
            D[i][j] = (ri.transpose() @ rj).sum()
            D[i][j] /= (np.linalg.norm(ri) * np.linalg.norm(rj))
    return D


if __name__ == "__main__":
    clean_trainging_set = remove_test_set(sample_user_ratings, sample_test_set)
    baseline = baseline_prediction(clean_trainging_set, lmda=0)
    print(baseline)
    error = clean_trainging_set - baseline
    print(error)
    cco = cosine_coefficient(error)
    print(cco)
