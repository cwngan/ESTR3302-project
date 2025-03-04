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


def average_ignore_none(data):
    """
    Find the average of a matrix ignoring None values.
    """
    res = 0
    count = 0
    for row in data:
        for col in row:
            if col is not None:
                res += col
                count += 1
    return res / count


def baseline_prediction(training_set, test_set):
    """
    Perform baseline prediction with given training set
    """
    for test in test_set:
        training_set[test[0]][test[1]] = None
    average_rating = average_ignore_none(training_set)
    # print(average_rating)

    A = []
    c = []

    for i, row in enumerate(training_set):
        for j, col in enumerate(row):
            if col is None:
                continue
            tmp = [0] * (len(training_set) + len(row))
            tmp[i] = 1
            tmp[len(training_set) + j] = 1
            A.append(tmp)
            c.append(col - average_rating)

    A = np.array(A)
    c = np.atleast_2d(c).transpose()
    # print(A)
    # print(c)
    AT = A.transpose()
    # print(AT)
    b = np.linalg.solve(AT @ A, AT @ c)
    output = np.zeros(shape=(len(training_set), len(training_set[0])))
    for i in range(len(training_set)):
        for j in range(len(training_set[0])):
            output[i][j] = (average_rating + b[i][0] +
                            b[len(training_set) + j][0])
    return output


if __name__ == "__main__":
    baseline = baseline_prediction(sample_user_ratings, sample_test_set)
    print(baseline)
