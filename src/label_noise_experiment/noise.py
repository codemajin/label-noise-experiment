"""Synthetic label noise injected into the training labels.

Every kind flips exactly round(rate * n) labels to a different class.

- symmetric: randomly chosen samples go to a uniformly random other class
- pair:      randomly chosen samples go to one fixed other class per class
- instance:  samples that a model trained on clean labels finds ambiguous are more
             likely to be chosen, and go to that model's runner-up class
             (closer to real annotation errors)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

NOISE_KINDS = ("symmetric", "pair", "instance")


def corrupt_labels(X, y, kind: str, rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    n_noisy = int(round(rate * len(y)))
    y_noisy = y.copy()

    if kind == "symmetric":
        idx = rng.choice(len(y), n_noisy, replace=False)
        shift = rng.integers(1, n_classes, size=n_noisy)
        y_noisy[idx] = (y[idx] + shift) % n_classes
    elif kind == "pair":
        # Permutation without fixed points: move each class to its neighbor in a random order
        order = rng.permutation(n_classes)
        target = np.empty(n_classes, dtype=int)
        target[order] = np.roll(order, -1)
        idx = rng.choice(len(y), n_noisy, replace=False)
        y_noisy[idx] = target[y[idx]]
    elif kind == "instance":
        proba = LogisticRegression(max_iter=2000).fit(X, y).predict_proba(X)
        uncertainty = 1.0 - proba[np.arange(len(y)), y] + 1e-3
        idx = rng.choice(
            len(y), n_noisy, replace=False, p=uncertainty / uncertainty.sum()
        )
        proba_other = proba[idx].copy()
        proba_other[np.arange(n_noisy), y[idx]] = -1.0
        y_noisy[idx] = proba_other.argmax(axis=1)
    else:
        raise ValueError(f"unknown noise kind: {kind}")

    return y_noisy
