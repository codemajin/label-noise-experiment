"""Run the experiment: train and evaluate every method for each condition
(dataset x noise kind x noise rate x seed)."""

import argparse
import itertools
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from .datasets import DATASETS, load_dataset, split_and_scale
from .methods import BASE_MODELS, iter_predictions, make_pseudo_splits, run_cleanlab
from .noise import NOISE_KINDS, corrupt_labels


def run_condition(model_name, dataset, noise, rate, seed, trusted_frac, oracle_fracs):
    warnings.filterwarnings("ignore")
    with threadpool_limits(1):
        X, y = load_dataset(dataset)
        X_tr, X_te, y_tr, y_te = split_and_scale(X, y, seed)
        y_noisy = corrupt_labels(X_tr, y_tr, noise, rate, seed)
        is_noisy = y_noisy != y_tr

        model = BASE_MODELS[model_name]()
        cl = run_cleanlab(model, X_tr, y_noisy, seed)
        splits = make_pseudo_splits(cl, y_noisy, trusted_frac, seed)
        # Thresholding the same uniform draws makes the smaller set a subset of the larger one
        u = np.random.default_rng(seed).random(len(y_tr))
        oracle_qualities = {round(f * 100): (u < f).astype(int) for f in oracle_fracs}

        cond = dict(model=model_name, dataset=dataset, noise=noise, rate=rate, seed=seed)

        flagged = cl.is_issue
        tp = np.sum(flagged & is_noisy)
        diag = [
            dict(
                cond,
                actual_noise_rate=is_noisy.mean(),
                cl_flag_rate=flagged.mean(),
                cl_precision=tp / max(flagged.sum(), 1),
                cl_recall=tp / max(is_noisy.sum(), 1),
                **{
                    f"{s}_{k}": v
                    for s, q in splits.items()
                    for k, v in (
                        ("trusted_frac", q.mean()),
                        ("trusted_noise", is_noisy[q == 1].mean()),
                        ("untrusted_noise", is_noisy[q == 0].mean()),
                    )
                },
            )
        ]

        rows = []
        for group, method, predict in iter_predictions(
            model, X_tr, y_noisy, y_tr, X_te, cl, splits, oracle_qualities
        ):
            t0 = time.perf_counter()
            try:
                y_pred = predict()
                acc = accuracy_score(y_te, y_pred)
                f1 = f1_score(y_te, y_pred, average="macro")
                err = ""
            except Exception as e:  # do not lose the whole condition when one method fails
                acc = f1 = np.nan
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            rows.append(
                dict(
                    cond,
                    group=group,
                    method=method,
                    accuracy=acc,
                    macro_f1=f1,
                    fit_sec=time.perf_counter() - t0,
                    error=err,
                )
            )
    return rows, diag


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(BASE_MODELS), default="lr")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--noise", nargs="+", default=list(NOISE_KINDS))
    p.add_argument("--rates", nargs="+", type=float, default=[0.2, 0.4])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--trusted-frac", type=float, default=0.3)
    p.add_argument("--oracle-fracs", nargs="+", type=float, default=[0.1, 0.3])
    p.add_argument("--n-jobs", type=int, default=-2)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    conditions = list(
        itertools.product(args.datasets, args.noise, args.rates, range(args.seeds))
    )
    print(f"[{args.model}] {len(conditions)} conditions")

    outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_condition)(
            args.model, d, n, r, s, args.trusted_frac, args.oracle_fracs
        )
        for d, n, r, s in conditions
    )

    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame([row for rows, _ in outputs for row in rows])
    diagnostics = pd.DataFrame([row for _, diag in outputs for row in diag])
    results.to_csv(out_dir / "results.csv", index=False)
    diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)
    print(f"saved to {out_dir}")
    n_err = (results["error"] != "").sum()
    if n_err:
        print(f"WARNING: {n_err} failed runs")
        print(results.loc[results["error"] != "", ["method", "error"]].drop_duplicates())


if __name__ == "__main__":
    main()
