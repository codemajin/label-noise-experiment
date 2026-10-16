"""Check the choice of HGB learning rate without using the test data.

Within the training split (70%) of each seed, compare the accuracy of stratified 5-fold
cross-validation on the true labels for each learning rate. This checks the stability of the
base model itself before any noise is injected, and never touches the test data.
"""

import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from threadpoolctl import threadpool_limits

from .datasets import DATASETS, load_dataset, split_and_scale

LEARNING_RATES = (0.1, 0.05)


def cv_accuracy(dataset, seed, lr):
    warnings.filterwarnings("ignore")
    with threadpool_limits(1):
        X, y = load_dataset(dataset)
        X_tr, _, y_tr, _ = split_and_scale(X, y, seed)
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=lr, random_state=0)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return dict(dataset=dataset, seed=seed, learning_rate=lr,
                    cv_accuracy=cross_val_score(model, X_tr, y_tr, cv=cv).mean())


def main():
    rows = Parallel(n_jobs=-2)(
        delayed(cv_accuracy)(d, s, lr)
        for d in DATASETS for s in range(5) for lr in LEARNING_RATES
    )
    df = pd.DataFrame(rows)
    df.to_csv("results/hgb/lr_check.csv", index=False)
    print(df.pivot_table(index="dataset", columns="learning_rate",
                         values="cv_accuracy", aggfunc=["mean", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
