import gym
from gym import spaces
from gym.spaces import MultiDiscrete
import numpy as np
from constants import DAYS, TIMES, SEASONS
from utils.context_extract import build_model_input_batch

class FilmRecEnv(gym.Env):
    """
    Defining the environment for film recommendations using only temporal context,
    but outputting a slate of recommendations (list of K films) per episode.

    State: one-hot day-of-week (7 dims) + one-hot time-of-day (4 dims) => 11 dims.

    Action: a slate of K film indices, represented via a MultiDiscrete space.

    Reward: a vector of length K reflecting user feedback for each recommended film.
    """
    def __init__(
            self,
            catalog_df,
            audience_model,
            memory_size: int = 5,
            rec_list_size: int = 3,
        ):
        super().__init__()
        self.catalog_df = catalog_df
        self.memory = []
        self.memory_size = memory_size
        self.K = rec_list_size
        self.n_films = len(catalog_df)
        self.audience_model = audience_model
        self.MAX_VIEWERS = 366

        # Temporal feature definitions
        self.DAYS = DAYS
        self.TIMES = TIMES
        self.SEASONS = SEASONS
        
        self.state = None  # will be set in reset()
        
        # ——— Action: slate of K film‐indices ———
        self.action_space = MultiDiscrete([self.n_films] * self.K)

        # ——— Observation: three integers ———
        # day_idx ∈ [0..6], time_idx ∈ [0..3], season_idx ∈ [0..len(SEASONS)-1]
        self.observation_space = MultiDiscrete([
            len(self.DAYS),
            len(self.TIMES),
            len(self.SEASONS)
        ])
        
        # Composite-reward weights
        default_w = {
            'accept':    1.0,
            'view':      0.5,
            'diversity': 0.2,
            'novelty':   0.2
        }
        self.w = default_w

        

    def build_state(self, context: dict) -> np.ndarray:

        """
        context = {'day': 'Fri', 'time': 'evening', 'season': 'Summer'}
        returns array([4, 2, 1], dtype=int32), for example
        """
        day_idx  = self.DAYS.index(context['day'])
        time_idx = self.TIMES.index(context['time'])
        season_idx = self.SEASONS.index(context['season'])
        return np.array([day_idx, time_idx, season_idx], dtype=np.int32)


    
    def reset(self, context: dict) -> np.ndarray:
        """
        Start a new recommendation episode with the given context and return state.
        """
        self.current_state = self.build_state(context)
        
        return self.current_state

    def step(self, slate: np.ndarray, selected_idx: int, context: dict):
        """
        slate:        np.ndarray of shape (K,) with film indices
        selected_idx: integer in [0..K-1] of which slot was accepted, or -1 if none
        """
        # 1) build batch of film features with context for the slate, as to predict expected viewership
        contexts = [context for _ in range(self.K)]  # same context for all slots
        input_slate = build_model_input_batch(contexts, movie_ids=slate)
        r_views = self.audience_model.predict(input_slate) / self.MAX_VIEWERS

        # 2) diversity & novelty bonuses
        r_div  = np.array([self._diversity_bonus(f) for f in slate])
        r_nov  = np.array([self._novelty_bonus(f)   for f in slate])

        # 3) accept signal
        r_accept = np.zeros(self.K, dtype=np.float32)
        if 0 <= selected_idx < self.K:
            r_accept[selected_idx] = 1.0
            chosen = int(slate[selected_idx])
            self.memory.append(chosen)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)

        # 4) composite reward
        rewards = (
            self.w['accept']    * r_accept +
            self.w['view']      * r_views   +
            self.w['diversity'] * r_div    +
            self.w['novelty']   * r_nov
        )

        done = True
        info = {
            'r_accept':    r_accept,
            'r_view':      r_views,
            'r_diversity': r_div,
            'r_novelty':   r_nov,
            'memory':      list(self.memory)
        }
        return self.state, rewards, done, info

    def _diversity_bonus(self, film_idx: int) -> float:
        # Example stub: reward if film’s genre is underrepresented
        return 0.0

    def _novelty_bonus(self, film_idx: int) -> float:
        # Example stub: 1 if film not in recent memory, else 0
        return 0.0
