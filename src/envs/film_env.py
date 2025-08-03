import gym
from gym import spaces
import numpy as np
from constants import DAYS, TIMES, MONTHS
from utils.context_extract import build_model_input_batch

class FilmRecEnvH(gym.Env):
    """
    Same hybrid encoding (one-hot day/time + cyclical month), but 
    step() returns only the “base” reward (accept + view + diversity + novelty)
    The competition bonus is omitted from training. Instead, you can call
    compute_competition_bonus(film_id, real_date) at inference time.

    State = 13 dims:
      • day_of_week (7-dim one-hot)
      • time_of_day (4-dim one-hot)
      • sin_month, cos_month (2 dims)

    Base-reward components (used in step()):
      • r_accept   (1 if clicked)
      • r_view     (normalized by MAX_VIEWERS)
      • r_diversity (stub)
      • r_novelty   (stub)

    At inference, use compute_competition_bonus(...) externally to add a bonus.
    """
    """
    A “heuristic” film-recommendation environment with:

    1) Hybrid one-hot encoding of (day_of_week, time_of_day, month) as a 23-dim
       feature vector:

         - day_of_week: 7-dim one-hot
         - time_of_day: 4-dim one-hot
         - month:       2-dim cyclical encoding

       State = [ day_onehot (7), time_onehot (4), sin_month, cos_month ] => 13 dims.

    2) A simple “competition” bonus: if a film is known to be airing on a
       competitor's channel in the same month, it receives a +1 bonus; otherwise 0.

    3) Memory/novelty/diversity stubs (left at 0).

    4) Composite reward = accept + view + diversity + novelty + competition.

    To use the competition heuristic, supply a dict
        competitor_schedule: dict[int, list[int]]
    mapping each film_idx → list of month_indices (0..11) when the competitor
    will air that film. If a given film_idx is not in the dict, it is assumed
    “never airs on the competitor” (→ no bonus).
    
    
    """

    def __init__(
        self,
        catalog_df,
        audience_model,
        competitor_schedule: dict[int, list[int]],
        memory_size: int = 5,
        rec_list_size: int = 3,
    ):
        super().__init__()
        self.catalog_df        = catalog_df
        self.audience_model    = audience_model
        self.competitor_schedule = competitor_schedule
        self.memory            = []
        self.memory_size       = memory_size
        self.K                 = rec_list_size
        self.n_films           = len(catalog_df)
        self.MAX_VIEWERS       = 366

        # Vocabularies (must match your constants.py)
        self.DAYS   = DAYS    # ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        self.TIMES  = TIMES   # ["morning","afternoon","evening","night"]
        self.MONTHS = MONTHS  # ["Jan","Feb",…,"Dec"]

        # ——— Action: slate of K film-indices ———
        self.action_space = spaces.MultiDiscrete([self.n_films] * self.K)

        # ——— Observation: 13-dim Box for hybrid one-hot + cyclical month ———
        #   7 dims for day_of_week one-hot
        #   4 dims for time_of_day one-hot
        #   2 dims for (sin_month, cos_month)
        self.observation_space = spaces.Box(
            low=-1.0,       # sine/cosine can be negative
            high=1.0,
            shape=(7 + 4 + 2,),  # = 13
            dtype=np.float32
        )

        # Composite-reward weights
        self.w = {
            'accept':      1.0,
            'view':        0.5,
            'diversity':   0.2,
            'novelty':     0.2,
            'competition': 0.4,
            'legal':       0.3,
        }

        self.current_state = None
        self._last_context = None

    def _encode_state(self, context: dict) -> np.ndarray:
        """
        Build a 13-dim vector:
          [ day_of_week_onehot (7),
            time_of_day_onehot (4),
            sin_month, cos_month (2) ]
        context must include:
          - 'day':   one of self.DAYS
          - 'time':  one of self.TIMES
          - 'month': one of self.MONTHS
        """
        # 1) day_of_week → 7-dim one-hot
        day_idx  = self.DAYS.index(context['day'])
        day_oh   = np.zeros(len(self.DAYS), dtype=np.float32)
        day_oh[day_idx] = 1.0

        # 2) time_of_day → 4-dim one-hot
        time_idx = self.TIMES.index(context['time'])
        time_oh  = np.zeros(len(self.TIMES), dtype=np.float32)
        time_oh[time_idx] = 1.0

        # 3) month → 2-dim cyclical
        month_idx = self.MONTHS.index(context['month'])  # 0..11
        theta = 2 * np.pi * month_idx / 12
        sin_m = np.sin(theta).astype(np.float32)
        cos_m = np.cos(theta).astype(np.float32)
        month_cyc = np.array([sin_m, cos_m], dtype=np.float32)

        # Concatenate → total shape (13,)
        return np.concatenate([day_oh, time_oh, month_cyc])

    def build_state(self, context: dict) -> np.ndarray:
        """
        Simply wraps _encode_state.
        """
        return self._encode_state(context)

    def reset(self, context: dict) -> np.ndarray:
        """
        context must contain:
          - 'day':   e.g. "Fri"
          - 'time':  e.g. "evening"
          - 'month': e.g. "Jul"
        Returns the 13-dim encoded state.
        """
        self._last_context  = context.copy()
        self.current_state  = self._encode_state(context)
        return self.current_state

    def step(self, slate: np.ndarray, selected_idx: int, context: dict):
        """
        slate:        np.ndarray of shape (K,) of film IDs
        selected_idx: int in [0..K-1] if clicked, else -1
        context:      same dict as passed to reset()

        Returns (state, rewards, done, info).
        """
        # 1) Viewership term
        contexts = [context] * self.K
        inp = build_model_input_batch(contexts, movie_ids=slate)
        r_views = (self.audience_model.predict(inp) / self.MAX_VIEWERS).astype(np.float32)

        # 2) Diversity & novelty stubs
        r_div = np.zeros(self.K, dtype=np.float32)
        r_nov = np.zeros(self.K, dtype=np.float32)

        # 3) Competition heuristic (binary same-month)
        month_idx = self.MONTHS.index(context['month'])
        r_comp = np.zeros(self.K, dtype=np.float32)
        for i, film_idx in enumerate(slate):
            months_list = self.competitor_schedule.get(film_idx, [])
            r_comp[i] = 1.0 if (month_idx in months_list) else 0.0

        # 4) Legal stub (always 0)
        r_legal = np.zeros(self.K, dtype=np.float32)

        # 5) Accept signal & memory
        r_accept = np.zeros(self.K, dtype=np.float32)
        if 0 <= selected_idx < self.K:
            r_accept[selected_idx] = 1.0
            chosen = int(slate[selected_idx])
            self.memory.append(chosen)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)

        # 6) Composite reward
        rewards = (
            self.w['accept']      * r_accept  +
            self.w['view']        * r_views   +
            self.w['diversity']   * r_div     +
            self.w['novelty']     * r_nov     +
            self.w['competition'] * r_comp
        ).astype(np.float32)

        done = True
        info = {
            'r_accept':     r_accept,
            'r_view':       r_views,
            'r_diversity':  r_div,
            'r_novelty':    r_nov,
            'r_competition':r_comp,
            'memory':       list(self.memory)
        }
        return self.current_state, rewards, done, info

    def _diversity_bonus(self, film_idx: int) -> float:
        return 0.0

    def _novelty_bonus(self, film_idx: int) -> float:
        return 0.0

    def _legal_bonus(self, film_idx: int) -> float:
        return 0.0
