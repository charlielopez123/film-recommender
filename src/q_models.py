import numpy as np
from typing import Sequence

class LinearQModel:
    """
    A linear bandit/Q-model that takes a MultiDiscrete state (list of integers)
    and one-hot-encodes each dimension before applying a weight matrix.
    """

    def __init__(
        self,
        state_nvec: Sequence[int],
        n_actions: int,
        lr: float = 0.1
    ):
        """
        state_nvec: sequence of N integers, where each is the number
                    of categories in that state dimension.
                    e.g. [7,4,4] for day(7), time(4), season(4)
        n_actions:  total number of arms (films)
        lr:          learning rate
        """
        self.state_nvec = list(state_nvec)
        self.n_actions  = n_actions
        self.lr         = lr

        # total feature dimension = sum of all one-hot dims
        self.feature_dim = sum(self.state_nvec)

        # Weight matrix W of shape (feature_dim, n_actions)
        self.W = np.zeros((self.feature_dim, self.n_actions), dtype=np.float32)

    def _encode_state(self, state: np.ndarray) -> np.ndarray:
        """
        Convert an integer state vector of shape (N,) into a float32
        one-hot feature vector of shape (feature_dim,).
        """
        features = []
        for val, dim in zip(state, self.state_nvec):
            one_hot = np.zeros(dim, dtype=np.float32)
            one_hot[int(val)] = 1.0
            features.append(one_hot)
        return np.concatenate(features)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        state: 1-D array of integers, length = len(self.state_nvec)
        returns:
          Q-values for all actions, shape = (n_actions,)
        """
        phi = self._encode_state(state)        # → (feature_dim,)
        return phi.dot(self.W)                 # → (n_actions,)

    def update(self, state: np.ndarray, action: int, reward: float):
        """
        One-step update for bandit/Q:
          Q(s,a) ← Q(s,a) + α (reward - Q(s,a))
        where Q(s,a) = φ(s)ᵀ W[:,action]
        """
        phi = self._encode_state(state)        # (feature_dim,)
        q_sa = phi.dot(self.W[:, action])      # scalar
        td_error = reward - q_sa
        # gradient wrt W[:,action] is phi
        self.W[:, action] += self.lr * td_error * phi
