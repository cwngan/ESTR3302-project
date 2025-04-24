import numpy as np


def generate_sample_data(users: int, items: int, factors: int):
    """
    Generate sample data for a user-item rating matrix.

    Optimized by o3-mini.

    Args:
        users (int): Number of users.
        items (int): Number of items.
        factors (int): Number of factors.
    """
    # 1) user_pref: U×F, item_perf: I×F
    user_pref = np.random.rand(users, factors) * 10.0 - 5
    user_sd = np.random.rand(users)  # U
    item_perf = np.random.rand(items, factors) * 10.0 - 5

    # 2) draw noise per (user, item, factor) in one shot
    #    shape: (users, items, factors)
    noise = np.random.normal(
        loc=user_pref[:, None, :],  # broadcast U×1×F
        scale=user_sd[:, None, None],  # broadcast U×1×1
        size=(users, items, factors),
    )

    # 3) weight by item_perf and sum over factors → (users, items)
    raw = (noise * item_perf[None, :, :]).sum(axis=2)
    user_bias = np.random.normal(0, 0.5, size=users)
    item_bias = np.random.normal(0, 1.0, size=items)
    raw += user_bias[:, None] + item_bias[None, :]

    # 4) min–max scale into [1, 5] and round
    mn, mx = raw.min(), raw.max()
    scaled = (raw - mn) / (mx - mn) * 4.0 + 1.0
    ratings = np.round(scaled).clip(1, 5)

    return {
        "user_pref": user_pref.tolist(),
        "user_sd": user_sd.tolist(),
        "item_perf": item_perf.tolist(),
        "ratings": ratings.astype(np.int64).tolist(),
    }
