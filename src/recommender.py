# src/recommender.py

import numpy as np
from envs.film_env import FilmRecEnv

# ——— Load your catalog & artifacts ———
catalog_df       = ...         # pandas DataFrame of your films
film_features    = ...         # np.ndarray shape (n_films, F)
max_viewers      = ...         # float, e.g. max observed in your logs
view_model_path  = "models/view_rf.joblib"

# ——— Instantiate environment ———
K   = 5
env = FilmRecEnv(
    n_films             = len(catalog_df),
    slate_size          = K,
    viewership_model_path = view_model_path,
    film_features       = film_features,
    max_viewers         = max_viewers,
    memory_size         = 5
)

# ——— Helper: derive slate from Q-model ———
def recommend_slate(q_model, state, memory, K):
    # 1) score all N films
    q_vals = q_model.predict(state)    # shape (n_films,)
    # 2) mask recently accepted
    q_vals[memory] = -np.inf
    # 3) pick top-K
    idx = np.argpartition(q_vals, -K)[-K:]
    return idx[np.argsort(-q_vals[idx])]











