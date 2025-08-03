# --- near the top of IL_training.py ---
import numpy as np

class ThompsonSampler:
    def __init__(self, dim, lam=1.0, sigma=1.0, prior_mean=None, seed = None):
        """
        dim   : feature dimension d
        lam   : ridge regularization
        sigma : assumed reward noise std
        """
        self.d = dim
        self.lam = lam
        self.sigma = sigma

        self.rng = np.random.default_rng(seed)

        # Prior mean m0
        if prior_mean is None:
            self.m0 = np.zeros(dim)
        else:
            self.m0 = np.array(prior_mean, dtype=float)

        # Initialize A = λ I, b = λ m0  (so that prior mean is A^{-1} b = m0)
        self.A = lam * np.eye(dim)
        self.b = lam * self.m0

    def warm_start(self, X_hist, R_hist):
        """
        X_hist : (N x d) matrix of past features
        R_hist : (N,) array of past rewards (0/1)
        """
        # accumulate A and b
        self.A += X_hist.T @ X_hist
        self.b += X_hist.T @ R_hist

    def sample_weights(self):
        """
        Draw w_tilde ~ N(mean = A^{-1} b, cov = σ^2 A^{-1})
        """
        mu = np.linalg.solve(self.A, self.b)
        cov = self.sigma**2 * np.linalg.inv(self.A)
        return np.random.multivariate_normal(mu, cov)

    def select_k(self, X_cands, K = 5):
        """
        X_cands : (M x d) matrix of candidate features
        returns index of the best candidate under a single TS draw
        """
        # 1) sample one hypothesis
        w_tilde = self.sample_weights()       # shape (d,)

        # 2) score every candidate
        scores = X_cands @ w_tilde            # shape (M,)

        # 3) pick top-K indices (fast via argpartition)
        idxs = np.argpartition(scores, -K)[-K:]
        # sort those K in descending score order
        topk = idxs[np.argsort(scores[idxs])[::-1]]
        return topk, scores[topk], w_tilde
    
    def update(self, x, r):
        """
        x : (d,) feature of the chosen movie
        r : observed reward (0 or 1)
        """
        self.A += np.outer(x, x)
        self.b += r * x
