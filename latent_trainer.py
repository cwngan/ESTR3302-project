import pickle

import numpy as np

from predictors.latent_factor import LatentFactorPredictor
from utils import get_test_set_matrix, remove_test_set, root_mean_square_error_entries
from utils.dataset import Dataset

full_dataset = Dataset(full=True)
training_data = remove_test_set(full_dataset.user_ratings, full_dataset.test_set)
test_data = get_test_set_matrix(full_dataset.user_ratings, full_dataset.test_set)

u, i = training_data.shape
k = 8
latent = LatentFactorPredictor(
    shape=training_data.shape,
    k=k,
    p=np.ones(shape=(k, u), dtype=np.float64),
    q=np.ones(shape=(k, i), dtype=np.float64),
    lmda=0.2,
)
print(f"{latent.p=}")
print(f"{latent.q=}")

latent.train(training_data=training_data, max_iterations=10000, tol=1e-4)

with open("models/latent_" + str(k), "wb") as f:
    pickle.dump(latent, f)

training_predictions = latent.predict(full_dataset.training_set)
test_predictions = latent.predict(full_dataset.test_set)
print(f"{training_predictions = }")
print(f"{test_predictions = }")
print(f"{latent.p = }")
print(f"{latent.q = }")


rmse_test = root_mean_square_error_entries(
    test_predictions, full_dataset.test_set, test_data
)
print(f"{rmse_test = }")
rmse_training = root_mean_square_error_entries(
    training_predictions, full_dataset.training_set, training_data
)
print(f"{rmse_training = }")
