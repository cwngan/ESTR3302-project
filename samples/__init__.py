import numpy as np


def generate_sample_data(users: int, items: int, factors: int):
    """
    Generate sample data with given parameters.

    Args:
        users (int): Number of users.
        items (int): Number of items.
        factors (int): Number of factors.
    """
    user_pref = np.random.rand(users, factors) * 10.0 - 5
    user_sd = np.random.rand(users)
    item_perf = np.random.rand(items, factors) * 10.0 - 5
    data = np.zeros((users, items), dtype=np.float64)
    for u in range(users):
        for i in range(items):
            score = 0
            for factor, fac_perf in enumerate(item_perf[i, :]):
                score += (
                    np.random.normal(loc=user_pref[u, factor], scale=user_sd[u])
                    * fac_perf
                )
            data[u, i] = score
    min_val = np.min(data)
    max_val = np.max(data)
    val_range = max_val - min_val
    data = (data - min_val) / val_range * 6.0
    return np.round(data.clip(min=1, max=5), 0)
