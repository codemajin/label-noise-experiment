"""Loading the experiment datasets and splitting them into train/test sets."""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype, load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DATASETS = ("digits", "synth", "covtype")

COVTYPE_N_SAMPLES = 12000


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a dataset as (X, y), where y holds integer labels starting at 0."""
    if name == "digits":
        d = load_digits()
        return d.data.astype(float), d.target
    if name == "synth":
        return make_classification(
            n_samples=8000,
            n_features=20,
            n_informative=10,
            n_redundant=4,
            n_classes=6,
            n_clusters_per_class=1,
            class_sep=1.0,
            random_state=0,
        )
    if name == "covtype":
        d = fetch_covtype(data_home=DATA_DIR)
        X, _, y, _ = train_test_split(
            d.data,
            d.target - 1,
            train_size=COVTYPE_N_SAMPLES,
            stratify=d.target,
            random_state=0,
        )
        return X.astype(float), y
    raise ValueError(f"unknown dataset: {name}")


def split_and_scale(X, y, seed: int, test_size: float = 0.3):
    """Stratified split followed by standardization fitted on the training split.
    Test labels are always clean."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te
