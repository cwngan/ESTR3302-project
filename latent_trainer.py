import pickle

import numpy as np

from predictors.latent_factor import LatentFactorPredictor
from utils import get_test_set_matrix, remove_test_set
from utils.dataset import Dataset

full_dataset = Dataset(full=True)
training_data = remove_test_set(full_dataset.user_ratings, full_dataset.test_set)
test_data = get_test_set_matrix(full_dataset.user_ratings, full_dataset.test_set)

u, i = training_data.shape
k = 2
latent = LatentFactorPredictor(
    shape=training_data.shape,
    k=k,
    p=np.ones(shape=(k, u), dtype=np.float64),
    q=np.ones(shape=(k, i), dtype=np.float64),
    lmda=0.2,
)
print(f"{latent.p=}")
print(f"{latent.q=}")

latent.train(training_data=training_data, max_iterations=100000, tol=1e-4)

with open("models/latent", "wb") as f:
    pickle.dump(latent, f)
