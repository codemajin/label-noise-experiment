"""Learning methods under comparison.

Every method sees only the noisy training labels; no human verification is involved.

1. baseline          : train on the noisy labels as they are
2. cl_prune          : drop the samples flagged by cleanlab and retrain (uniform weights)
   cl_cleanlearning  : cleanlab's CleanLearning (after pruning, weight class k by 1/P(y~=k|y=k))
3. bq/<split>/<algo> : build pseudo trusted/untrusted sets from cleanlab's output and pass
                       them to biquality-learn
4. cl_weight_* / cl_feature : use cleanlab's label quality score as a weight or a feature

As references we also report oracle_clean (trained on clean labels; the upper bound if every
label were fixed by hand) and oracle_trusted<pct>/irbl (a random subset fixed by hand and used
as the trusted set).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from bqlearn.density_ratio import KKMM, KPDR
from bqlearn.irbl import IRBL
from bqlearn.irlnl import IRLNL
from bqlearn.plugin import PluginCorrection
from bqlearn.unbiased import LossCorrection
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BASE_MODELS: dict[str, Callable[[], object]] = {
    "lr": lambda: LogisticRegression(max_iter=2000),
    "hgb": lambda: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, random_state=0
    ),
}


@dataclass
class CleanlabResult:
    pred_probs: np.ndarray  # out-of-sample predicted probabilities from cross-validation
    is_issue: np.ndarray  # whether cleanlab flagged the label as an error
    self_confidence: np.ndarray  # predicted probability of the given label (label quality score)
    normalized_margin: np.ndarray


def run_cleanlab(model, X, y_noisy, seed: int) -> CleanlabResult:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred_probs = cross_val_predict(
        clone(model), X, y_noisy, cv=cv, method="predict_proba"
    )
    return CleanlabResult(
        pred_probs=pred_probs,
        is_issue=find_label_issues(y_noisy, pred_probs),
        self_confidence=get_label_quality_scores(
            y_noisy, pred_probs, method="self_confidence"
        ),
        normalized_margin=get_label_quality_scores(
            y_noisy, pred_probs, method="normalized_margin"
        ),
    )


def make_pseudo_splits(
    cl: CleanlabResult, y_noisy, trusted_frac: float, seed: int
) -> dict[str, np.ndarray]:
    """Build sample_quality (1 = trusted, 0 = untrusted) from cleanlab's result.

    - naive  : every unflagged sample is trusted, every flagged one untrusted
    - topconf: per class, the top trusted_frac of unflagged samples by quality score are trusted
    - random : per class, a random trusted_frac of unflagged samples are trusted
    """
    rng = np.random.default_rng(seed)
    splits = {"naive": (~cl.is_issue).astype(int)}

    for name in ("topconf", "random"):
        quality = np.zeros(len(y_noisy), dtype=int)
        for c in np.unique(y_noisy):
            in_class = np.flatnonzero(y_noisy == c)
            candidates = in_class[~cl.is_issue[in_class]]
            if len(candidates) == 0:  # keep at least one trusted sample even if a whole class is flagged
                candidates = in_class
            k = max(1, int(round(trusted_frac * len(in_class))))
            k = min(k, len(candidates))
            if name == "topconf":
                chosen = candidates[np.argsort(-cl.self_confidence[candidates])[:k]]
            else:
                chosen = rng.choice(candidates, k, replace=False)
            quality[chosen] = 1
        splits[name] = quality

    return splits


def biquality_algorithms(model) -> dict[str, Callable[[], object]]:
    lr = LogisticRegression(max_iter=2000)
    return {
        "trusted_only": lambda: clone(model),  # train on the trusted set only (for comparison)
        "irbl": lambda: IRBL(clone(model), clone(model)),
        "kpdr": lambda: KPDR(clone(model), pdr_estimator=clone(lr)),
        "kkmm": lambda: KKMM(clone(model), batch_size=1000),
        "glc": lambda: LossCorrection(clone(model), transition_matrix="gold"),
        "plugin": lambda: PluginCorrection(clone(model), transition_matrix="gold"),
        "irlnl": lambda: IRLNL(clone(model), clone(model), transition_matrix="gold"),
    }


def iter_predictions(
    model,
    X_tr,
    y_noisy,
    y_clean,
    X_te,
    cl: CleanlabResult,
    splits: dict[str, np.ndarray],
    oracle_qualities: dict[int, np.ndarray],
) -> Iterator[tuple[str, str, Callable[[], np.ndarray]]]:
    """Yield (group, method, function returning test-set predictions) in turn.

    oracle_qualities maps {percentage: mask of random samples assumed to be fixed by hand}
    (used for the reference methods).
    """

    def fit_predict(X, y, X_test, **fit_kw):
        return clone(model).fit(X, y, **fit_kw).predict(X_test)

    yield "ref", "oracle_clean", lambda: fit_predict(X_tr, y_clean, X_te)
    # The intended use of biquality-learn: fix the labels of a random subset by hand and trust it
    for pct, q in oracle_qualities.items():
        y_mixed = np.where(q == 1, y_clean, y_noisy)
        yield "ref", f"oracle_trusted{pct}/irbl", lambda q=q, y_mixed=y_mixed: (
            IRBL(clone(model), clone(model))
            .fit(X_tr, y_mixed, sample_quality=q)
            .predict(X_te)
        )
    yield "1", "baseline", lambda: fit_predict(X_tr, y_noisy, X_te)

    keep = ~cl.is_issue
    yield "2", "cl_prune", lambda: fit_predict(X_tr[keep], y_noisy[keep], X_te)
    # CleanLearning estimates the label issues and the noise matrix itself from the same pred_probs
    yield "2", "cl_cleanlearning", lambda: (
        CleanLearning(clone(model))
        .fit(X_tr, y_noisy, pred_probs=cl.pred_probs)
        .predict(X_te)
    )

    for split_name, quality in splits.items():
        for algo_name, factory in biquality_algorithms(model).items():
            if algo_name == "trusted_only":
                if split_name == "naive":  # naive trusted_only is identical to cl_prune
                    continue
                t = quality == 1
                pred = lambda t=t: fit_predict(X_tr[t], y_noisy[t], X_te)
            else:
                pred = lambda f=factory, q=quality: (
                    f().fit(X_tr, y_noisy, sample_quality=q).predict(X_te)
                )
            yield "3", f"bq/{split_name}/{algo_name}", pred

    for score_name in ("self_confidence", "normalized_margin"):
        w = getattr(cl, score_name)
        yield "4", f"cl_weight/{score_name}", lambda w=w: fit_predict(
            X_tr, y_noisy, X_te, sample_weight=w
        )

    # Append the quality score as a feature and, at inference, feed "most reliable label" (= 1).
    # The label quality score is unavailable at inference, so we condition on its maximum instead.
    X_tr_aug = np.column_stack([X_tr, cl.self_confidence])
    X_te_aug = np.column_stack([X_te, np.ones(len(X_te))])
    yield "4", "cl_feature", lambda: fit_predict(X_tr_aug, y_noisy, X_te_aug)
