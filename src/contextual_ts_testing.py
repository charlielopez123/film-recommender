from datetime import datetime, timedelta
import zoneinfo
from contextual_thompson import ContextualThompsonSampler
import random
from constants import all_reward_feature_names

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple
import math

def create_date_list(days_ahead = 15):
    tz = zoneinfo.ZoneInfo("Europe/Zurich")
    today = datetime.now(tz).date()
    dates = [today + timedelta(days=i) for i in range(0, days_ahead)]
    return ([d.isoformat() for d in dates])

hours = [9, 13, 15, 20, 22, 24]

def marginal_weight_distribution_cts(
    cts,
    env,
    all_feature_names: Sequence[str] = all_reward_feature_names,
    context_features = None, 
    num_samples: int = 1000,
    bins: int = 100,
    show: bool = False,
) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Draw marginal posterior distributions of contextual weights w_t for random contexts,
    all with the same axis scales for easy comparison.
    """
    # 1) Sample many weight vectors
    all_w = []
    dates = create_date_list()
    for _ in range(num_samples):
        if context_features is None:
            date = random.choice(dates)
            hour = random.choice(hours)
            context = env.create_context_from_date(date, hour)
            context_f, _ = env.get_context_features(context)
        else:
            context_f = context_features
        all_w.append(cts.thompson_sample_w(context_f))

    all_w = np.vstack(all_w)  # shape = (num_samples, d)
    num_signals = all_w.shape[1]

    # 2) Determine global x‐limits
    x_min, x_max = all_w.min(), all_w.max()
    # 3) Precompute common bin edges
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    # 4) Determine global y‐limit by seeing the largest bin count across features
    max_count = 0
    for i in range(num_signals):
        counts, _ = np.histogram(all_w[:, i], bins=bin_edges)
        max_count = max(max_count, counts.max())

    # 5) Prepare feature names
    if len(all_feature_names) >= num_signals:
        feature_names = list(all_feature_names[:num_signals])
    else:
        feature_names = [f"w[{i}]" for i in range(num_signals)]

    # 6) Create subplots
    cols = min(4, num_signals)
    rows = math.ceil(num_signals / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    # 7) Plot each histogram with the same edges and y‐limit
    for i in range(num_signals):
        ax = axes[i]
        ax.hist(all_w[:, i], bins=bin_edges, edgecolor="black", alpha=0.7)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, max_count * 1.05)  # slight headroom
        ax.set_title(f"{feature_names[i]} marginal distribution")
        ax.set_xlabel(f"{feature_names[i]} value")
        ax.set_ylabel("count")

    # 8) Remove unused axes
    for j in range(num_signals, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes[:num_signals], all_w


def marginal_weight_distribution_cts_2(
    cts,
    env,
    all_feature_names: Sequence[str],
    context_features = None, 
    num_samples: int = 1000,
    bins: int = 30,
    show: bool = False,
) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Draw marginal posterior distributions of contextual weights w_t for random contexts.

    Args:
        cts: contextual Thompson sampler with method thompson_sample_w(context_feat)
        env: environment providing context construction
        all_feature_names: list of length d with names for each weight dimension
        hours: list of possible hours to sample contexts from
        num_samples: how many Thompson draws / contexts to sample
        bins: histogram bins per feature
        show: whether to call plt.show()

    Returns:
        fig: matplotlib Figure object
        axes: array of axes used (one per feature)
        all_w: array shape (num_samples, d) of sampled weight vectors
    """
    all_w = []
    dates = create_date_list()  # assumed to exist in scope
    if context_features is None:
        for _ in range(num_samples):
            date = random.choice(dates)
            hour = random.choice(hours)
            context = env.create_context_from_date(date, hour)
            context_f, _ = env.get_context_features(context)
            all_w.append(cts.thompson_sample_w(context_f))
    else:
        for _ in range(num_samples):
            date = random.choice(dates)
            hour = random.choice(hours)
            all_w.append(cts.thompson_sample_w(context_features))

    all_w = np.vstack(all_w)  # shape = (num_samples, d)
    d = all_w.shape[1]

    # Prepare feature name list, fallback to indexed names if mismatch
    if len(all_feature_names) >= d:
        feature_names = list(all_feature_names[:d])
    else:
        feature_names = [f"w[{i}]" for i in range(d)]

    cols = min(4, d)
    rows = math.ceil(d / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)  # flatten in case of single row/col

    for i in range(d):
        ax = axes[i]
        ax.hist(all_w[:, i], bins=bins, edgecolor="black", alpha=0.7)
        ax.set_title(f"{feature_names[i]} marginal distribution")
        ax.set_xlabel(f"{feature_names[i]} value")
        ax.set_ylabel("count")

    # Remove unused axes
    for j in range(d, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes[:d], all_w




def marginal_weight_distribution_cts3(cts: ContextualThompsonSampler, context_f: np.ndarray, num_samples: int = 1000):
    # ————————————————————————————————
    # 1) Draw many posterior samples
    # ————————————————————————————————
    # Assume `ts` is your ThompsonSampler instance
    # and ts.sample_weights() returns (w_sample, mu, cov)
    all_w = np.vstack([
        cts.thompson_sample_w(context_f)
        for _ in range(num_samples)
    ])  # shape = (num_samples, d)

    # ————————————————————————————————
    # 2) Plot marginal histograms for each weight
    # ————————————————————————————————
    for i in range(len(all_w), 0):
        plt.figure()
        plt.hist(all_w[:, i], bins=30)
        plt.title(f"Weight #{i} marginal distribution")
        plt.xlabel(f"w[{i}] value")
        plt.ylabel("count")
        plt.show()
