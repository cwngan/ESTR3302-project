import numpy as np
from scipy.sparse import csr_matrix


def generate_sample_data(users: int, items: int, factors: int):
    """
    Generate sample data for a user-item rating matrix.

    The function generates a matrix in which only 2% of the entries are filled with ratings,
    and the resulting rating matrix is returned as a sparse csr_matrix.
    This optimized version computes ratings only for sampled indices to avoid creating huge dense matrices.

    Args:
        users (int): Number of users.
        items (int): Number of items.
        factors (int): Number of factors.
    """
    # Generate latent factors and biases
    user_pref = np.random.normal(0, 1.5, size=(users, factors))
    user_sd = np.random.rand(users)
    item_perf = np.random.normal(0, 1.5, size=(items, factors))
    user_bias = np.random.normal(0, 0.5, size=users)
    item_bias = np.random.normal(0, 1.0, size=items)

    # Determine sample indices for 2% non-zero entries
    total_entries = users * items
    sample_size = int(total_entries * 0.02)
    indices = np.random.choice(total_entries, sample_size, replace=False)
    rows, cols = np.unravel_index(indices, (users, items))

    # Compute raw ratings only for sampled indices
    noise = np.random.normal(
        loc=user_pref[rows, :],
        scale=user_sd[rows][:, None],
        size=(sample_size, factors),
    )
    raw_values = (
        np.sum(noise * item_perf[cols, :], axis=1) + user_bias[rows] + item_bias[cols]
    )

    # Min–max scale into [1, 5] and round.
    mn, mx = raw_values.min(), raw_values.max()
    scaled = (raw_values - mn) / (mx - mn) * 4.0 + 0.5
    ratings_sampled = (np.round(scaled * 2) / 2).clip(0.5, 5)

    # Create sparse matrix from sampled ratings.
    ratings = csr_matrix((ratings_sampled, (rows, cols)), shape=(users, items))

    return {
        "user_pref": user_pref.tolist(),
        "user_sd": user_sd.tolist(),
        "item_perf": item_perf.tolist(),
        "ratings": ratings,
    }
