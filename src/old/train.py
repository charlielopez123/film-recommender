import random
import numpy as np
from envs.film_env import FilmRecEnvH
from old.q_models import LinearQModel
import pandas as pd
import pickle
from constants import DAYS, TIMES, SEASONS, MONTHS
from config import settings

# ——— HYPERPARAMETERS ———
N_EPISODES    = 10_000
EPSILON_START = 0.5
EPSILON_END   = 0.05
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / N_EPISODES

def sample_context():
    return {
        "day":    random.choice(DAYS),
        "time":   random.choice(TIMES),
        "month": random.choice(MONTHS)
    }

# ——— SETUP ———
catalog_df = pd.read_pickle("data/catalog_df.pkl")            # load your DataFrame
with open(settings.aud_model_dir/"rf_model.pkl", "rb") as f:    # load your frozen viewership model
    audience_model = pickle.load(f)

env = FilmRecEnvH(
    catalog_df=catalog_df,
    audience_model=audience_model,
    memory_size=5,
    rec_list_size=3
)
state_nvec = env.observation_space.nvec  # e.g. [7, 4, 4] for day(7), time(4), season(4)
n_actions = env.n_films

q_model = LinearQModel(state_nvec, n_actions, lr=0.1)

# ——— TRAINING LOOP ———
epsilon = EPSILON_START
for ep in range(1, N_EPISODES+1):
    # 1) Sample a random temporal context
    context = sample_context()
    env.state = env.reset(context)

    # 2) Compute Q‐values and mask memory
    q_vals = q_model.predict(env.state)            # shape (n_films,)
    if env.memory:
        q_vals[env.memory] = -np.inf

    # 3) Build slate by ε‐greedy
    if random.random() < epsilon:
        slate = random.sample(range(env.n_films), env.K)
    else:
        # top‐K by Q‐value
        idx = np.argpartition(q_vals, -env.K)[-env.K:] # get indices from top-K but not sorted, more efficient
        slate = idx[np.argsort(-q_vals[idx])] # sort the top-K indices by Q-value

    # 4) Simulate user choice: pick slot with highest viewership estimate
    #    (or -1 for “no click” if you want)
    selected_slot = -1 # default to no click
    
    # 5) Step env and get rewards
    _, rewards, done, info = env.step(slate, selected_slot, context)

    # 6) Update Q‐model for each slot as an independent bandit
    for slot_idx, movie_id in enumerate(slate):
        r = rewards[slot_idx]
        q_model.update(env.state, movie_id, r)

    # 7) Decay epsilon
    epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)

    # 8) Logging every so often
    if ep % 1000 == 0:
        avg_reward = rewards.mean()
        print(f"Episode {ep}/{N_EPISODES}  ε={epsilon:.3f}  last_avg_r={avg_reward:.3f}")

# ——— Save your trained Q‐model ———
import joblib
joblib.dump(q_model, "models/q_model.joblib")
