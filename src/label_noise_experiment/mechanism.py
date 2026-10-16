"""Mechanism analysis. For each condition we record:

- for each way of building trusted/untrusted (pseudo: naive/random/topconf; genuine: oracle10/oracle30)
  - the mean diagonal of the transition matrix estimated by GLC (gold)
  - the fraction of correct labels in the untrusted set (GLC estimates the matrix by evaluating a
    classifier trained on U over T, so the estimate can reflect the noise rate of U rather than
    that of the whole dataset)
- for each pseudo split, the mean IRBL weight on untrusted samples whose labels are actually
  wrong / correct
- for comparison, the mean diagonal of the true transition matrix of the whole dataset
  (an unweighted average over classes)
"""

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bqlearn.irbl import IRBL
from bqlearn.unbiased import LossCorrection
from joblib import Parallel, delayed
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from .datasets import DATASETS, load_dataset, split_and_scale
from .methods import BASE_MODELS, make_pseudo_splits, run_cleanlab
from .noise import NOISE_KINDS, corrupt_labels

ORACLE_FRACS = (0.1, 0.3)


def run_condition(model_name, dataset, noise, rate, seed, trusted_frac):
    warnings.filterwarnings("ignore")
    with threadpool_limits(1):
        X, y = load_dataset(dataset)
        X_tr, _, y_tr, _ = split_and_scale(X, y, seed)
        n_classes = int(y.max()) + 1
        y_noisy = corrupt_labels(X_tr, y_tr, noise, rate, seed)
        is_noisy = y_noisy != y_tr

        model = BASE_MODELS[model_name]()
        cl = run_cleanlab(model, X_tr, y_noisy, seed)
        pseudo = make_pseudo_splits(cl, y_noisy, trusted_frac, seed)

        # (split name, sample_quality, labels used for training). oracle restores trusted labels to the truth
        u = np.random.default_rng(seed).random(len(y_tr))
        settings = [(name, q, y_noisy) for name, q in pseudo.items()]
        for f in ORACLE_FRACS:
            q = (u < f).astype(int)
            settings.append((f"oracle{round(f * 100)}", q, np.where(q == 1, y_tr, y_noisy)))

        t_true = np.zeros((n_classes, n_classes))
        np.add.at(t_true, (y_tr, y_noisy), 1)
        t_true /= t_true.sum(axis=1, keepdims=True)
        row = dict(model=model_name, dataset=dataset, noise=noise, rate=rate, seed=seed,
                   true_diag=np.diag(t_true).mean())

        for name, q, labels in settings:
            untrusted = q == 0
            row[f"{name}_u_clean"] = 1.0 - is_noisy[untrusted].mean()
            glc = LossCorrection(clone(model), transition_matrix="gold")
            glc.fit(X_tr, labels, sample_quality=q)
            row[f"{name}_glc_diag"] = np.diag(glc.transition_matrix_).mean()
            if name in pseudo:
                w = IRBL(clone(model), clone(model)).fit(X_tr, labels, sample_quality=q).sample_weight_
                row[f"{name}_irbl_w_noisy"] = w[untrusted & is_noisy].mean()
                row[f"{name}_irbl_w_clean"] = w[untrusted & ~is_noisy].mean()
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(BASE_MODELS), default="lr")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--trusted-frac", type=float, default=0.3)
    p.add_argument("--n-jobs", type=int, default=-2)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    conditions = itertools.product(DATASETS, NOISE_KINDS, [0.2, 0.4], range(args.seeds))
    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_condition)(args.model, d, n, r, s, args.trusted_frac)
        for d, n, r, s in conditions
    )
    df = pd.DataFrame(rows)
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "mechanism.csv", index=False)
    print(df.drop(columns=["seed"]).groupby(["noise", "rate"]).mean(numeric_only=True).round(2).T)
    print(f"saved to {out_dir / 'mechanism.csv'}")


if __name__ == "__main__":
    main()
