import numpy as np
from typing import Optional
from typing import Union
import pickle
from pathlib import Path

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softplus(x):
    # stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def inverse_softplus(y):
    # Numerically stable inverse of softplus; y must be > 0
    eps = 1e-6
    y = np.clip(y, eps, None)
    return np.log(np.expm1(y))  # expm1(y) = exp(y) - 1

class ContextualThompsonSampler:
    def __init__(
        self,
        num_signals: int,
        context_dim: int,
        lr: float = 0.01,
        reg: float = 1e-1,
        expl_scale: float = 1.0,
        ema_decay: float = 0.95,
        max_grad_norm = 5.0,
        h0 = 1e-3,
        tau = 2,
        alpha = 0.3,
        seed: Optional[int] = 42,
    ):
        """
        num_signals: length of s_{i,t} (number of base signals)
        context_dim: length of context vector c_t
        lr: learning rate for MAP updates of U,b
        reg: Gaussian prior strength (L2) on U,b
        expl_scale: scaling of Thompson sampling noise (higher -> more exploration)
        ema_decay: for running diagonal precision approximation (like RMSProp)
        """
        self.num_signals = num_signals
        self.context_dim = context_dim
        self.lr = lr
        self.reg = reg
        self.expl_scale = expl_scale
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm        # clip threshold
        self.h0 = h0
        self.epsilon = 1e-8 
        self.tau = tau
        self.alpha = alpha

        rng = np.random.default_rng(seed)
        # initialize U and b small
        self.U = rng.standard_normal((num_signals, context_dim)) * 0.01
        self.b = np.zeros(num_signals)

        # running squared-grad estimate for approximate precision (diagonal Hessian)
        self.h_U = np.ones_like(self.U) * self.h0  # avoid divide-by-zero
        self.h_b = np.ones_like(self.b) * self.h0

        self.rng = rng

    def _compute_w(self, U, b, c_t, tau=2.0, alpha= 0.2):
        raw = U @ c_t + b
        raw = raw / tau
        ex = np.exp(raw - raw.max())
        p = ex / ex.sum()
        w = (1-alpha)*p + alpha/len(p)
        return w, raw

    
    
    def _compute_w_softmax(self, U, b, c_t, tau=1.0):
        raw = U @ c_t + b            # (num_signals,)
        #raw = np.clip(raw, -10, 10)  # bounds to avoid extreme exponentials
        scaled = raw / tau
        ex = np.exp(scaled - np.max(scaled))
        w = ex / (ex.sum() + 1e-12)
        return w, raw

    def _compute_w_softplus(self, U, b, c_t):
        """
        context to positive weights (no normalization here; can add if desired)
        """
        raw = U @ c_t + b  # shape (num_signals,)
        w = softplus(raw)  # positive
        #w = raw
        w = w / (w.sum() + 1e-8)      # normalize to sum to 1
        return w, raw  # return raw for gradients

    def thompson_sample_w(self, c_t):
        """
        Sample a plausible w_t via Thompson draw over (U,b) using diagonal approx.
        """
        # sample U_tilde and b_tilde from approximate Gaussian posterior around current MAP
        # variance approximated as inverse of h (plus small eps); scale by expl_scale
        eps = 1e-8
        std_U = np.sqrt(self.expl_scale / (self.h_U + eps))
        std_b = np.sqrt(self.expl_scale / (self.h_b + eps))

        U_tilde = self.U + self.rng.normal(size=self.U.shape) * std_U
        b_tilde = self.b + self.rng.normal(size=self.b.shape) * std_b

        w_tilde, _ = self._compute_w(U_tilde, b_tilde, c_t)
        return w_tilde  # positive weights

    def score_candidates(self, c_t: np.ndarray, S_matrix: np.ndarray, K: int = 5):
        """
        c_t: (context_dim,) context vector
        S_matrix: (M, num_signals) base signal vectors for M candidates
        returns: top-K indices, their scores, sampled weight vector w_tilde, raw thetas
        """
        w_tilde = self.thompson_sample_w(c_t)  # context weights
        scores = S_matrix @ w_tilde  # shape (M,), linear utility
        idxs = np.argpartition(scores, -K)[-K:]
        topk = idxs[np.argsort(scores[idxs])[::-1]]
        return topk, scores[topk], w_tilde, scores
    
    def update(
    self,
    c_t: np.ndarray,
    s_chosen: np.ndarray,
    r: float,
    y_signals: Optional[np.ndarray] = None,
    match_weight: float = 0.1,
    lr: Optional[float] = None,
    ):
        """
        Single-step MAP update with:
        - logistic loss on (picked vs not),
        - plus optional squared‐error signal‐matching loss if y_signals provided.

        c_t        : (context_dim,)
        s_chosen   : (num_signals,)
        r          : reward {0,1}
        y_signals  : optional binary vector length num_signals of curator‐cited signals
        match_weight: weight on the match loss term
        lr         : override learning rate
        """
        # Forward pass through softmax+smoothing compute_w
        w, raw = self._compute_w(self.U, self.b, c_t)  # w=smoothed-softmax(raw), raw=U@c_t + b
        
        # ---- 1) Logistic reward loss ----
        theta = w.dot(s_chosen)
        p = sigmoid(theta)
        dL_dtheta = (p - r)  # scalar
        grad_w_logit = dL_dtheta * s_chosen          # (d,)
        # derivative of w wrt raw: Jacobian of smoothed softmax.
        # For simplicity, we treat it as: dp_i/draw_j = 
        #    (1-alpha)*( softmax_j * (δ_{ij}-softmax_i) )
        prob = np.exp(raw/ self.tau - np.max(raw/self.tau))
        prob /= prob.sum()
        # Jacobian of softmax
        J = np.diag(prob) - np.outer(prob, prob)
        J *= (1 - self.alpha) / self.tau  # incorporate smoothing mix & temperature
        grad_raw_logit = J.T @ grad_w_logit  # (d,)

        # ---- 2) Optional signal‐matching loss ----
        if y_signals is not None and y_signals.sum() > 0:
            # normalize signals to a distribution
            y_norm = y_signals / (y_signals.sum() + 1e-12)  # (d,)
            # invert softplus‐smoothmax to get target raw: we approximate
            # target_raw = inverse_softplus(y_norm)  # if using softplus
            # but here with softmax mix, we can instead directly match w:
            # L_match = 0.5 * match_weight * ||w - y_norm||^2
            grad_w_match = match_weight * (w - y_norm)    # (d,)
            # backprop through same Jacobian J
            grad_raw_match = J.T @ grad_w_match           # (d,)
        else:
            grad_raw_match = 0

        # ---- 3) Combine gradients ----
        grad_raw = grad_raw_logit + grad_raw_match  # total dL/d(raw)

        # ---- 4) Gradients w.r.t U and b ----
        grad_U = np.outer(grad_raw, c_t)            # (d, p)
        grad_b = grad_raw                          # (d,)

        # Regularization
        grad_U += self.reg * self.U
        grad_b += self.reg * self.b

        # Gradient clipping
        norm_U = np.linalg.norm(grad_U)
        if norm_U > self.max_grad_norm:
            grad_U *= (self.max_grad_norm / (norm_U + 1e-12))
        norm_b = np.linalg.norm(grad_b)
        if norm_b > self.max_grad_norm:
            grad_b *= (self.max_grad_norm / (norm_b + 1e-12))

        # ---- 5) Update curvature EMA ----
        self.h_U = self.ema_decay * self.h_U + (1 - self.ema_decay) * (grad_U**2)
        self.h_b = self.ema_decay * self.h_b + (1 - self.ema_decay) * (grad_b**2)

        # ---- 6) Gradient step ----
        if lr is None:
            lr = self.lr
        self.U -= lr * grad_U
        self.b -= lr * grad_b


    def update_old(self, c_t: np.ndarray, s_chosen: np.ndarray, r: float, lr = None):
        """
        Single-step MAP update with logistic loss for chosen candidate.
        c_t: (context_dim,)
        s_chosen: (num_signals,) base signal vector of chosen movie
        r: reward {0,1}
        """
        print(f'hyperparams: ema_decay = {self.ema_decay}')
        # forward
        w, raw = self._compute_w(self.U, self.b, c_t)  # (num_signals,)
        theta = w.dot(s_chosen)  # scalar
        p = sigmoid(theta)

        # logistic loss gradient: dL/dtheta = p - r
        dL_dtheta = p - r  # shape ()

        # gradient w.r.t w: (p - r) * s_chosen
        grad_w = dL_dtheta * s_chosen  # shape (num_signals,)

        # derivative of softplus is sigmoid(raw)
        grad_raw = grad_w * sigmoid(raw)  # shape (num_signals,)

        # gradients for U and b
        grad_U = np.outer(grad_raw, c_t)  # (num_signals, context_dim)
        grad_b = grad_raw  # (num_signals,)

        # add regularization (gradient of 0.5 * reg * ||param||^2)
        grad_U += self.reg * self.U
        grad_b += self.reg * self.b

        # gradient clipping (per-matrix / vector)
        norm_U = np.linalg.norm(grad_U)
        if norm_U > self.max_grad_norm:
            grad_U = grad_U * (self.max_grad_norm / (norm_U + 1e-12))
                       
           
        norm_b = np.linalg.norm(grad_b)
        if norm_b > self.max_grad_norm:
            grad_b = grad_b * (self.max_grad_norm / (norm_b + 1e-12))
            
        
        # update running diagonal approximation (like Laplace/RMSProp)
        self.h_U = self.ema_decay * self.h_U + (1 - self.ema_decay) * (grad_U ** 2)
        self.h_b = self.ema_decay * self.h_b + (1 - self.ema_decay) * (grad_b ** 2)

        if lr is None:
            lr = self.lr
        # gradient step (using raw gradient, not scaled by h here; adaptive behavior comes from sampling noise)
        self.U -= lr * grad_U
        self.b -= lr * grad_b
        print(f'grad_U: {grad_U.mean()}, grad_b: {grad_b.mean()}')



    def warm_start(self, contexts: Union[np.ndarray, float], signals: Union[np.ndarray, float], rewards: Union[np.ndarray, float], epochs: int = 1, lr = 1e-3):
        """
        contexts: (N, context_dim)
        signals:  (N, num_signals)  -- s_{i,t} for the chosen items in history
        rewards:  (N,) binary rewards 0/1
        """
        # many samples are passed when warm starting we want to minimize the noise compared to the online exploring where the actual signals are factored in the choices
        old_ema_decay = self.ema_decay
        old_expl_scale = self.expl_scale
        self.expl_scale = self.expl_scale
        self.ema_decay = 1-1e-4
        if isinstance(rewards, float):
            c_t = contexts
            s_i = signals
            r = rewards
            self.update(c_t, s_i, r, lr = lr)
        else:
            n = len(rewards)
            for _ in range(epochs):
                perm = self.rng.permutation(n)
                for idx in perm:
                    c_t = contexts[idx]
                    s_i = signals[idx]
                    r = rewards[idx]
                    self.update(c_t, s_i, r, lr = lr)

        # reset hyperparamters to old ones
        self.ema_decay = old_ema_decay
        self.expl_scale = old_expl_scale
                    


    def set_uniform_target_weights(self, w_target):
        """
        Initialize so that for all contexts, the normalized softplus weights start as w_target.
        w_target: (d,) positive vector (will be normalized internally)
        """
        w_target = np.asarray(w_target, dtype=float)
        # normalize target to sum to 1 (convex combination)
        w_target = w_target / (w_target.sum() + 1e-12)

        # Set U to zero so context has no effect initially
        self.U = np.zeros_like(self.U)

        # Invert softplus to get b such that softplus(b) ∝ w_target
        # Because we normalize after softplus, exact proportionality is enough.
        self.b = inverse_softplus(w_target)

    def save(self, path: Union[str, Path]):
        """
        Serialize the sampler to disk:
          - <path>.npz      : U, b, h_U, h_b, and hyperparameters
          - <path>.rng.pkl  : RNG bit‐generator state
        """
        path = Path(path)
        # Save numeric state
        np.savez(
            path.with_suffix(path.suffix or ".npz"),
            U=self.U,
            b=self.b,
            h_U=self.h_U,
            h_b=self.h_b,
            lr=np.array(self.lr),
            reg=np.array(self.reg),
            expl_scale=np.array(self.expl_scale),
            ema_decay=np.array(self.ema_decay),
            max_grad_norm=np.array(self.max_grad_norm),
            h0=np.array(self.h0),
            alpha=np.array(self.alpha),
            tau = np.array(self.tau)
        )
        # Save RNG state
        rng_state = self.rng.bit_generator.state
        with open(path.with_suffix(".rng.pkl"), "wb") as f:
            pickle.dump(rng_state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ContextualThompsonSampler":
        """
        Load a sampler previously saved with `.save()`.
        """
        path = Path(path)
        npz = path.with_suffix(path.suffix or ".npz")
        data = np.load(npz)
        # Reconstruct
        sampler = cls(
            num_signals=data["U"].shape[0],
            context_dim=data["U"].shape[1],
            lr=float(data["lr"]),
            reg=float(data["reg"]),
            expl_scale=float(data["expl_scale"]),
            ema_decay=float(data["ema_decay"]),
            max_grad_norm=float(data["max_grad_norm"]),
            h0=float(data["h0"]),
            tau=float(data["tau"]),
            alpha=float(data["alpha"]),
            seed=None,  # we'll restore RNG explicitly
        )
        sampler.U = data["U"]
        sampler.b = data["b"]
        sampler.h_U = data["h_U"]
        sampler.h_b = data["h_b"]
        # Restore RNG
        rng_file = path.with_suffix(".rng.pkl")
        if rng_file.exists():
            with open(rng_file, "rb") as f:
                state = pickle.load(f)
            sampler.rng = np.random.default_rng()
            sampler.rng.bit_generator.state = state
        return sampler


    