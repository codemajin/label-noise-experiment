"""Control experiment: biquality-learn given a genuine clean trusted set instead of a pseudo one.

The labels of a random oracle_frac of the samples are restored to the clean labels and trusted.
The same random draws as oracle_trusted<pct>/irbl in run.py select the same samples.
This checks whether methods that break with pseudo trusted sets work under their intended assumptions.
"""

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

from .datasets import DATASETS, load_dataset, split_and_scale
from .methods import BASE_MODELS, biquality_algorithms
from .noise import NOISE_KINDS, corrupt_labels

ALGOS = ("trusted_only", "irbl", "kpdr", "glc", "plugin", "irlnl")


def run_condition(model_name, dataset, noise, rate, seed, oracle_frac):
    warnings.filterwarnings("ignore")
    with threadpool_limits(1):
        X, y = load_dataset(dataset)
        X_tr, X_te, y_tr, y_te = split_and_scale(X, y, seed)
        y_noisy = corrupt_labels(X_tr, y_tr, noise, rate, seed)
        quality = (np.random.default_rng(seed).random(len(y_tr)) < oracle_frac).astype(int)
        y_mixed = np.where(quality == 1, y_tr, y_noisy)

        model = BASE_MODELS[model_name]()
        algos = biquality_algorithms(model)
        cond = dict(model=model_name, dataset=dataset, noise=noise, rate=rate, seed=seed)
        rows = [
            dict(cond, method="baseline",
                 accuracy=accuracy_score(y_te, clone(model).fit(X_tr, y_noisy).predict(X_te)))
        ]
        for name in ALGOS:
            if name == "trusted_only":
                t = quality == 1
                est = clone(model).fit(X_tr[t], y_tr[t])
            else:
                est = algos[name]().fit(X_tr, y_mixed, sample_quality=quality)
            rows.append(dict(cond, method=f"oracle/{name}",
                             accuracy=accuracy_score(y_te, est.predict(X_te))))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(BASE_MODELS), default="lr")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--oracle-frac", type=float, default=0.1)  # output: oracle_check_<pct>.csv
    p.add_argument("--n-jobs", type=int, default=-2)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    conditions = itertools.product(DATASETS, NOISE_KINDS, [0.2, 0.4], range(args.seeds))
    outputs = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_condition)(args.model, d, n, r, s, args.oracle_frac)
        for d, n, r, s in conditions
    )
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for rows in outputs for r in rows]).to_csv(
        out_dir / f"oracle_check_{round(args.oracle_frac * 100)}.csv", index=False
    )
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
