"""Write the paper's tables (.tex), the number macros used in the text, and the figure (.pdf) from results/ into latex/.

Tables are written in English (latex/tables/en/) and Japanese (latex/tables/ja/);
the number macros (latex/numbers.tex) and the figure are shared by both papers.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .summarize import COND, add_delta, plot_delta

MODELS = ("lr", "hgb")
MODEL_LABEL = {"lr": "LR", "hgb": "HGB"}
NOISE_ORDER = ("symmetric", "pair", "instance")
RATES = (0.2, 0.4)
ORACLE_PCTS = (10, 30)
LANGS = ("en", "ja")
BQ_ALGOS = ("irbl", "kpdr", "kkmm", "glc", "plugin", "irlnl")

# Order of methods in the tables (by group). "ref" is translated through LABELS.
METHOD_GROUPS = [
    ("ref", ["oracle_clean", "oracle_trusted10/irbl", "oracle_trusted30/irbl"]),
    ("(1)", ["baseline"]),
    ("(2)", ["cl_prune", "cl_cleanlearning"]),
    ("(3) naive", [f"bq/naive/{a}" for a in BQ_ALGOS]),
    ("(3) random", [f"bq/random/{a}" for a in ("trusted_only", *BQ_ALGOS)]),
    ("(3) topconf", [f"bq/topconf/{a}" for a in ("trusted_only", *BQ_ALGOS)]),
    ("(4)", ["cl_weight/self_confidence", "cl_weight/normalized_margin", "cl_feature"]),
]

KEY_METHODS = [
    "oracle_clean", "oracle_trusted10/irbl", "oracle_trusted30/irbl",
    "cl_prune", "cl_cleanlearning", "bq/naive/irbl", "bq/random/irbl", "bq/topconf/irbl",
    "bq/naive/kpdr", "cl_weight/self_confidence", "cl_feature", "bq/random/glc", "bq/naive/glc",
]

# Pairs shown in the paired-comparison table, with short names for the text macros
PAIRS = [
    ("NaiveIrblVsPrune", "bq/naive/irbl", "cl_prune"),
    ("NaiveIrblVsCL", "bq/naive/irbl", "cl_cleanlearning"),
    ("CLVsPrune", "cl_cleanlearning", "cl_prune"),
    ("RandomIrblVsRandomT", "bq/random/irbl", "bq/random/trusted_only"),
    ("NaiveIrblVsRandomIrbl", "bq/naive/irbl", "bq/random/irbl"),
    ("RandomIrblVsTopIrbl", "bq/random/irbl", "bq/topconf/irbl"),
    ("RandomTVsTopT", "bq/random/trusted_only", "bq/topconf/trusted_only"),
    ("WeightVsPrune", "cl_weight/self_confidence", "cl_prune"),
    ("FeatureVsPrune", "cl_feature", "cl_prune"),
    ("OracleTenVsNaiveIrbl", "oracle_trusted10/irbl", "bq/naive/irbl"),
    ("OracleThirtyVsNaiveIrbl", "oracle_trusted30/irbl", "bq/naive/irbl"),
    ("OracleThirtyVsOracleTen", "oracle_trusted30/irbl", "oracle_trusted10/irbl"),
]

# Table header labels per language
LABELS = {
    "en": {
        "ref": "Ref.", "group": "Group", "method": "Method", "acc": "Acc.", "win": "Win",
        "baseline_acc": r"baseline acc.\ (\%)", "detection": "cleanlab detection",
        "flag": "Flagged", "precision": "Prec.", "recall": "Recall",
        "model": "Model", "noise": "Noise", "rate": "Rate",
        "method_a": "Method A", "method_b": "Method B", "a_win": "A win",
        "algorithm": "Algorithm", "glc_diag": "Mean diagonal estimated by GLC",
        "u_clean": "Clean rate of U", "true": "True", "noisy": "Noisy", "clean": "Clean",
        "dataset": "Dataset", "learning_rate": "Learning rate",
    },
    "ja": {
        "ref": "参照", "group": "群", "method": "手法", "acc": "精度", "win": "勝率",
        "baseline_acc": r"baseline 精度 (\%)", "detection": "cleanlab の検出",
        "flag": "指摘率", "precision": "適合率", "recall": "再現率",
        "model": "モデル", "noise": "ノイズ", "rate": "率",
        "method_a": "手法 A", "method_b": "手法 B", "a_win": "A 勝率",
        "algorithm": "アルゴリズム", "glc_diag": "GLC が推定した対角平均",
        "u_clean": "U の正ラベル率", "true": "真値", "noisy": "誤", "clean": "正",
        "dataset": "データセット", "learning_rate": "学習率",
    },
}


def tt(method: str) -> str:
    return r"\texttt{" + method.replace("_", r"\_") + "}"


def num(x: float, signed: bool = False, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s.replace("-", "$-$")


def pct(rate: float) -> str:
    return f"{round(rate * 100)}\\%"


def tabular(colspec: str, body: list[str]) -> str:
    return "\n".join([rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *body,
                      r"\bottomrule", r"\end{tabular}", ""])


def write(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path}")


def delta_vs_baseline(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Add the difference from the baseline of the same condition (delta) to oracle_check results."""
    base = baseline.set_index(["model", *COND])["accuracy"]
    out = df.set_index(["model", *COND])
    out = out.assign(delta=(out["accuracy"] - base.reindex(out.index)) * 100)
    return out.reset_index()


# ---------------------------------------------------------------- tables


def overall_table(df: pd.DataFrame, lang: str) -> str:
    L = LABELS[lang]
    g = df.groupby(["model", "method"])
    acc = g["accuracy"].mean() * 100
    delta = g["delta"].mean()
    ci = g["delta"].sem() * 1.96
    win = g["delta"].apply(lambda d: (d > 0).mean() * 100)

    # Bold the method with the largest delta for each model (excluding references)
    non_ref = [m for _, ms in METHOD_GROUPS[1:] for m in ms]
    best = {m: delta[m].loc[non_ref].idxmax() for m in MODELS}

    body = [
        " & ".join(["", ""] + [rf"\multicolumn{{3}}{{c}}{{{MODEL_LABEL[m]}}}" for m in MODELS]) + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(MODELS))),
        f"{L['group']} & {L['method']} & "
        + " & ".join([rf"{L['acc']} & $\Delta$ (95\% CI) & {L['win']}"] * len(MODELS)) + r" \\",
        r"\midrule",
    ]
    for gi, (group, methods) in enumerate(METHOD_GROUPS):
        if gi:
            body.append(r"\addlinespace")
        for i, m in enumerate(methods):
            cells = [L.get(group, group) if i == 0 else "", tt(m)]
            for model in MODELS:
                d = num(delta[model, m], signed=True)
                if m == best[model]:
                    d = r"\textbf{" + d + "}"
                c = "" if m == "baseline" else rf" {{\scriptsize$\pm${ci[model, m]:.2f}}}"
                cells += [num(acc[model, m]), d + c, "--" if m == "baseline" else f"{win[model, m]:.0f}"]
            body.append(" & ".join(cells) + r" \\")
    return tabular("ll" + "rrr" * len(MODELS), body)


def breakdown_table(df: pd.DataFrame, methods: list[str], by: str, lang: str) -> str:
    """Delta by (noise kind x noise rate) or by dataset, in one panel per model.
    Each panel starts with the absolute baseline accuracy (%)."""
    L = LABELS[lang]
    if by == "noise":
        cols = [(n, r) for n in NOISE_ORDER for r in RATES]
        key = ["noise", "rate"]
        header = [
            " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in NOISE_ORDER) + r" \\",
            " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(3)),
            " & " + " & ".join(pct(r) for _, r in cols) + r" \\",
        ]
    else:
        cols = [(d,) for d in sorted(df["dataset"].unique())]
        key = ["dataset"]
        header = [" & " + " & ".join(c[0] for c in cols) + r" \\"]

    body = []
    for mi, model in enumerate(MODELS):
        d = df[df["model"] == model]
        t = d.groupby(["method", *key])["delta"].mean()
        a = d.groupby(["method", *key])["accuracy"].mean() * 100
        if mi:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textbf{{{MODEL_LABEL[model]}}}}} \\")
        body += [*header, r"\midrule"]
        body.append(rf"\textit{{{L['baseline_acc']}}} & " + " & ".join(num(a[("baseline", *c)]) for c in cols) + r" \\")
        body.append(r"\addlinespace")
        for m in methods:
            body.append(tt(m) + " & " + " & ".join(num(t.get((m, *c), np.nan), signed=True) for c in cols) + r" \\")
    return tabular("l" + "r" * len(cols), body)


def diagnostics_table(diag: pd.DataFrame, lang: str) -> str:
    L = LABELS[lang]
    cols = [
        ("cl_flag_rate", L["flag"]), ("cl_precision", L["precision"]), ("cl_recall", L["recall"]),
        ("naive_trusted_noise", "T"), ("naive_untrusted_noise", "U"),
        ("random_trusted_noise", "T"), ("random_untrusted_noise", "U"),
        ("topconf_trusted_noise", "T"), ("topconf_untrusted_noise", "U"),
    ]
    g = diag.groupby(["model", "noise", "rate"])[[c for c, _ in cols]].mean() * 100
    body = [
        rf" & & & \multicolumn{{3}}{{c}}{{{L['detection']}}} & \multicolumn{{2}}{{c}}{{naive}} & \multicolumn{{2}}{{c}}{{random}} & \multicolumn{{2}}{{c}}{{topconf}} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12}",
        f"{L['model']} & {L['noise']} & {L['rate']} & " + " & ".join(h for _, h in cols) + r" \\",
        r"\midrule",
    ]
    for mi, model in enumerate(MODELS):
        if mi:
            body.append(r"\addlinespace")
        first = True
        for noise in NOISE_ORDER:
            for rate in RATES:
                row = g.loc[(model, noise, rate)]
                cells = [MODEL_LABEL[model] if first else "", noise, pct(rate)]
                cells += [f"{row[c]:.1f}" for c, _ in cols]
                body.append(" & ".join(cells) + r" \\")
                first = False
    return tabular("lll" + "r" * len(cols), body)


def paired_diffs(df: pd.DataFrame) -> dict:
    """{(model, key): (mean, ci, win)}"""
    p = df.pivot_table(index=["model", *COND], columns="method", values="accuracy") * 100
    out = {}
    for key, a, b in PAIRS:
        for model in MODELS:
            d = p.loc[model, a] - p.loc[model, b]
            out[model, key] = (d.mean(), 1.96 * d.sem(), (d > 0).mean() * 100)
    return out


def paired_table(df: pd.DataFrame, lang: str) -> str:
    L = LABELS[lang]
    diffs = paired_diffs(df)
    body = [
        " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{MODEL_LABEL[m]}}}" for m in MODELS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(MODELS))),
        f"{L['method_a']} & {L['method_b']} & "
        + " & ".join([rf"A$-$B (95\% CI) & {L['a_win']}"] * len(MODELS)) + r" \\",
        r"\midrule",
    ]
    for key, a, b in PAIRS:
        cells = [tt(a), tt(b)]
        for model in MODELS:
            mean, ci, win = diffs[model, key]
            cells += [num(mean, signed=True) + rf" {{\scriptsize$\pm${ci:.2f}}}", f"{win:.0f}"]
        body.append(" & ".join(cells) + r" \\")
    return tabular("ll" + "rr" * len(MODELS), body)


def oracle_table(oracle: pd.DataFrame, df: pd.DataFrame, lang: str) -> str:
    """Delta with pseudo trusted sets (naive / random) and genuine ones (oracle 10% / 30%)."""
    L = LABELS[lang]
    o_mean = oracle.groupby(["model", "frac", "method"])["delta"].mean()
    d_mean = df.groupby(["model", "method"])["delta"].mean()
    heads = ["naive", "random", *(f"oracle{p}" for p in ORACLE_PCTS)]
    n = len(heads)
    body = [
        " & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{MODEL_LABEL[m]}}}" for m in MODELS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + n * i}-{1 + n * (i + 1)}}}" for i in range(len(MODELS))),
        f"{L['algorithm']} & " + " & ".join([" & ".join(heads)] * len(MODELS)) + r" \\",
        r"\midrule",
    ]
    for a in ("trusted_only", "irbl", "kpdr", "glc", "plugin", "irlnl"):
        cells = [tt(a)]
        for model in MODELS:
            naive = d_mean[model, "cl_prune"] if a == "trusted_only" else d_mean[model, f"bq/naive/{a}"]
            cells.append(num(naive, signed=True) + (r"$^\dagger$" if a == "trusted_only" else ""))
            cells.append(num(d_mean[model, f"bq/random/{a}"], signed=True))
            cells += [num(o_mean.get((model, p, f"oracle/{a}"), np.nan), signed=True) for p in ORACLE_PCTS]
        body.append(" & ".join(cells) + r" \\")
    return tabular("l" + "r" * n * len(MODELS), body)


def oracle_breakdown_table(oracle: pd.DataFrame, lang: str) -> str:
    """Delta with a genuine trusted set by dataset x noise (averaged over rates and seeds)."""
    L = LABELS[lang]
    datasets = sorted(oracle["dataset"].unique())
    cols = [(d, n) for d in datasets for n in NOISE_ORDER]
    t = oracle.groupby(["model", "method", "frac", "dataset", "noise"])["delta"].mean()
    short = {"symmetric": "sym", "pair": "pair", "instance": "inst"}
    body = [
        " & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{d}}}" for d in datasets) + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(datasets))),
        f"{L['algorithm']} & trusted & " + " & ".join(short[n] for _, n in cols) + r" \\",
        r"\midrule",
    ]
    for mi, model in enumerate(MODELS):
        body.append((r"\midrule" if mi else "") + rf"\multicolumn{{{len(cols) + 2}}}{{l}}{{\textbf{{{MODEL_LABEL[model]}}}}} \\")
        for a in ("irbl", "glc", "plugin", "irlnl"):
            for i, p in enumerate(ORACLE_PCTS):
                cells = [tt(a) if i == 0 else "", f"{p}\\%"]
                cells += [num(t.get((model, f"oracle/{a}", p, d, n), np.nan), signed=True, digits=1) for d, n in cols]
                body.append(" & ".join(cells) + r" \\")
    return tabular("ll" + "r" * len(cols), body)


def mechanism_glc_table(mech: pd.DataFrame, lang: str) -> str:
    L = LABELS[lang]
    glc_cols = ["naive", "random", "topconf", "oracle10", "oracle30"]
    u_cols = ["naive", "random", "oracle10"]
    g = mech.groupby(["model", "noise", "rate"]).mean(numeric_only=True)
    body = [
        rf" & & & \multicolumn{{5}}{{c}}{{{L['glc_diag']}}} & \multicolumn{{3}}{{c}}{{{L['u_clean']}}} \\",
        r"\cmidrule(lr){4-8} \cmidrule(lr){9-11}",
        f"{L['model']} & {L['noise']} & {L['rate']} & {L['true']} & " + " & ".join(glc_cols + u_cols) + r" \\",
        r"\midrule",
    ]
    for mi, model in enumerate(MODELS):
        if mi:
            body.append(r"\addlinespace")
        first = True
        for noise in NOISE_ORDER:
            for rate in RATES:
                r = g.loc[(model, noise, rate)]
                cells = [MODEL_LABEL[model] if first else "", noise, pct(rate), f"{r['true_diag']:.2f}"]
                cells += [f"{r[f'{c}_glc_diag']:.2f}" for c in glc_cols]
                cells += [f"{r[f'{c}_u_clean']:.2f}" for c in u_cols]
                body.append(" & ".join(cells) + r" \\")
                first = False
    return tabular("lll" + "r" * (1 + len(glc_cols) + len(u_cols)), body)


def mechanism_irbl_table(mech: pd.DataFrame, lang: str) -> str:
    L = LABELS[lang]
    splits = ["naive", "random", "topconf"]
    g = mech.groupby(["model", "noise", "rate"]).mean(numeric_only=True)
    body = [
        " & & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{s}}}" for s in splits) + r" \\",
        " ".join(rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(splits))),
        f"{L['model']} & {L['noise']} & {L['rate']} & "
        + " & ".join([f"{L['noisy']} & {L['clean']}"] * len(splits)) + r" \\",
        r"\midrule",
    ]
    for mi, model in enumerate(MODELS):
        if mi:
            body.append(r"\addlinespace")
        first = True
        for noise in NOISE_ORDER:
            for rate in RATES:
                r = g.loc[(model, noise, rate)]
                cells = [MODEL_LABEL[model] if first else "", noise, pct(rate)]
                for s in splits:
                    cells += [f"{r[f'{s}_irbl_w_noisy']:.2f}", f"{r[f'{s}_irbl_w_clean']:.2f}"]
                body.append(" & ".join(cells) + r" \\")
                first = False
    return tabular("lll" + "rr" * len(splits), body)


def hgb_lr_table(lr_check: pd.DataFrame, lang: str) -> str:
    L = LABELS[lang]
    g = lr_check.groupby(["dataset", "learning_rate"])["cv_accuracy"].agg(["mean", "min", "max"]) * 100
    rates = sorted(lr_check["learning_rate"].unique(), reverse=True)
    body = [
        f"{L['dataset']} & " + " & ".join(f"{L['learning_rate']} {r}" for r in rates) + r" \\",
        r"\midrule",
    ]
    for d in sorted(lr_check["dataset"].unique()):
        cells = [d] + [f"{g.loc[(d, r), 'mean']:.1f} ({g.loc[(d, r), 'min']:.1f}--{g.loc[(d, r), 'max']:.1f})" for r in rates]
        body.append(" & ".join(cells) + r" \\")
    return tabular("l" + "r" * len(rates), body)


# ---------------------------------------------------------------- number macros for the text


def macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}" + "\n"


def macros(df, diag, oracle, mech, lr_check) -> str:
    out = []
    n_cond = len(df[df["model"] == MODELS[0]][COND].drop_duplicates())
    out.append(macro("NCond", str(n_cond)))

    method_keys = [
        ("Prune", "cl_prune"), ("CleanLearning", "cl_cleanlearning"), ("NaiveIRBL", "bq/naive/irbl"),
        ("Oracle", "oracle_clean"), ("OracleTen", "oracle_trusted10/irbl"),
        ("OracleThirty", "oracle_trusted30/irbl"), ("Feature", "cl_feature"),
        ("WeightSC", "cl_weight/self_confidence"), ("NaiveGLC", "bq/naive/glc"),
        ("NaivePlugin", "bq/naive/plugin"), ("NaiveIRLNL", "bq/naive/irlnl"),
    ]
    diffs = paired_diffs(df)
    for model in MODELS:
        M = MODEL_LABEL[model]
        d = df[df["model"] == model]
        means = d.groupby("method")["delta"].mean()
        for key, m in method_keys:
            out.append(macro(f"D{M}{key}", num(means[m], signed=True)))
        out.append(macro(f"Acc{M}Baseline", f"{d[d['method'] == 'baseline']['accuracy'].mean() * 100:.2f}"))
        fit = d.groupby("method")["fit_sec"].mean()
        out.append(macro(f"Time{M}Ratio", f"{fit['bq/naive/irbl'] / fit['cl_prune']:.1f}"))
        for key, *_ in PAIRS:
            mean, ci, win = diffs[model, key]
            out.append(macro(f"P{M}{key}", num(mean, signed=True)))
            out.append(macro(f"P{M}{key}CI", f"{ci:.2f}"))
            out.append(macro(f"P{M}{key}Win", f"{win:.0f}"))
        o = oracle[oracle["model"] == model].groupby(["frac", "method"])["delta"].mean()
        for a in ("glc", "plugin", "irlnl", "irbl", "trusted_only"):
            for p in ORACLE_PCTS:
                name = {10: "Ten", 30: "Thirty"}[p]
                out.append(macro(f"O{M}{a.replace('_', '').upper()}{name}", num(o[p, f"oracle/{a}"], signed=True)))
        # Size of the naive trusted set (% of the training data)
        t = diag[diag["model"] == model].groupby(["dataset", "noise", "rate"])["naive_trusted_frac"].mean() * 100
        out.append(macro(f"NaiveTSizeMin{M}", f"{t.min():.0f}"))
        out.append(macro(f"NaiveTSizeMax{M}", f"{t.max():.0f}"))

        mm = mech[mech["model"] == model].mean(numeric_only=True)
        for name, col in [
            ("TrueDiag", "true_diag"), ("NaiveDiag", "naive_glc_diag"), ("RandomDiag", "random_glc_diag"),
            ("TopDiag", "topconf_glc_diag"), ("OracleTenDiag", "oracle10_glc_diag"),
            ("OracleThirtyDiag", "oracle30_glc_diag"), ("NaiveUClean", "naive_u_clean"),
            ("RandomUClean", "random_u_clean"), ("OracleTenUClean", "oracle10_u_clean"),
            ("WNoisy", "naive_irbl_w_noisy"), ("WClean", "naive_irbl_w_clean"),
            ("RandomWNoisy", "random_irbl_w_noisy"), ("RandomWClean", "random_irbl_w_clean"),
            ("TopWNoisy", "topconf_irbl_w_noisy"), ("TopWClean", "topconf_irbl_w_clean"),
        ]:
            out.append(macro(f"Mech{M}{name}", f"{mm[col]:.2f}"))

    inst = diag[diag["noise"] == "instance"].groupby(["model", "dataset", "rate"])["cl_recall"].mean() * 100
    out.append(macro("InstRecallMin", f"{inst.min():.0f}"))
    out.append(macro("InstRecallMax", f"{inst.max():.0f}"))
    un = diag.groupby(["model", "dataset", "noise", "rate"])["naive_untrusted_noise"].mean() * 100
    out.append(macro("NaiveUntrustedMin", f"{un.min():.0f}"))
    out.append(macro("NaiveUntrustedMax", f"{un.max():.0f}"))

    lc = lr_check.groupby(["dataset", "learning_rate"])["cv_accuracy"].mean() * 100
    out.append(macro("CovLRHigh", f"{lc['covtype', 0.1]:.1f}"))
    out.append(macro("CovLRLow", f"{lc['covtype', 0.05]:.1f}"))
    return "".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--latex", type=Path, default=Path("latex"))
    args = p.parse_args(argv)
    R = args.results

    results = pd.concat(pd.read_csv(R / m / "results.csv") for m in MODELS)
    diag = pd.concat(pd.read_csv(R / m / "diagnostics.csv") for m in MODELS)
    df = add_delta(results)
    baseline = results[results["method"] == "baseline"]
    oracle = pd.concat(
        pd.read_csv(R / m / f"oracle_check_{p}.csv").assign(frac=p) for m in MODELS for p in ORACLE_PCTS
    )
    oracle = delta_vs_baseline(oracle[oracle["method"] != "baseline"], baseline)
    mech = pd.concat(pd.read_csv(R / m / "mechanism.csv") for m in MODELS)
    lr_check = pd.read_csv(R / "hgb" / "lr_check.csv")

    all_methods = [m for _, ms in METHOD_GROUPS for m in ms if m != "baseline"]
    for lang in LANGS:
        tables = args.latex / "tables" / lang
        write(tables / "overall.tex", overall_table(df, lang))
        write(tables / "by_noise_key.tex", breakdown_table(df, KEY_METHODS, "noise", lang))
        write(tables / "by_noise_all.tex", breakdown_table(df, all_methods, "noise", lang))
        write(tables / "by_dataset_all.tex", breakdown_table(df, all_methods, "dataset", lang))
        write(tables / "diagnostics.tex", diagnostics_table(diag, lang))
        write(tables / "paired.tex", paired_table(df, lang))
        write(tables / "oracle.tex", oracle_table(oracle, df, lang))
        write(tables / "oracle_breakdown.tex", oracle_breakdown_table(oracle, lang))
        write(tables / "mechanism_glc.tex", mechanism_glc_table(mech, lang))
        write(tables / "mechanism_irbl.tex", mechanism_irbl_table(mech, lang))
        write(tables / "hgb_lr.tex", hgb_lr_table(lr_check, lang))
    write(args.latex / "numbers.tex", macros(df, diag, oracle, mech, lr_check))

    (args.latex / "figures").mkdir(parents=True, exist_ok=True)
    plot_delta(df, args.latex / "figures" / "delta_vs_baseline.pdf")
    print("wrote figures/delta_vs_baseline.pdf")


if __name__ == "__main__":
    main()
