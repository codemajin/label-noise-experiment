"""Aggregate results/<model>/*.csv into results/summary.md and a figure."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COND = ["dataset", "noise", "rate", "seed"]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def add_delta(results: pd.DataFrame) -> pd.DataFrame:
    """Add the accuracy difference (points) from the baseline of the same condition."""
    base = results.loc[results["method"] == "baseline", ["model", *COND, "accuracy"]]
    base = base.rename(columns={"accuracy": "baseline_acc"})
    df = results.merge(base, on=["model", *COND])
    df["delta"] = (df["accuracy"] - df["baseline_acc"]) * 100
    return df


def overall_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = df.groupby(["model", *COND])["accuracy"].rank(ascending=False)
    g = df.groupby(["model", "group", "method"])
    t = pd.DataFrame(
        {
            "acc": g["accuracy"].mean() * 100,
            "Δ vs baseline": g["delta"].mean(),
            "win rate": g["delta"].apply(lambda d: (d > 0).mean() * 100),
            "mean rank": g["rank"].mean(),
            "fit sec": g["fit_sec"].mean(),
        }
    )
    return t.reset_index().sort_values(["model", "Δ vs baseline"], ascending=[True, False])


def breakdown_table(df: pd.DataFrame, model: str, by: list[str]) -> pd.DataFrame:
    d = df[df["model"] == model]
    t = d.pivot_table(index="method", columns=by, values="delta", aggfunc="mean")
    return t.loc[t.mean(axis=1).sort_values(ascending=False).index]


def plot_delta(df: pd.DataFrame, path: Path):
    present = set(df["model"])
    models = [m for m in ("lr", "hgb") if m in present] + sorted(present - {"lr", "hgb"})
    stats = (
        df[df["method"] != "baseline"]
        .groupby(["model", "method"])["delta"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    order = (
        stats.groupby("method")["mean"].mean().sort_values(ascending=True).index.tolist()
    )

    fig, axes = plt.subplots(
        1, len(models), figsize=(5.2 * len(models), 0.32 * len(order) + 1.6),
        sharey=True, sharex=True, facecolor=SURFACE,
    )
    axes = [axes] if len(models) == 1 else axes
    for ax, model in zip(axes, models):
        s = stats[stats["model"] == model].set_index("method").reindex(order)
        y = range(len(order))
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.errorbar(
            s["mean"], y, xerr=1.96 * s["sem"], fmt="o", color=SERIES,
            markersize=6, elinewidth=2, capsize=0,
        )
        ax.set_yticks(list(y), order, fontsize=9, color=TEXT_PRIMARY)
        ax.set_title(f"base model: {model}", fontsize=11, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("Δ accuracy vs baseline (pt, mean ± 95% CI)", fontsize=9,
                      color=TEXT_SECONDARY)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fmt(t: pd.DataFrame) -> str:
    return t.to_markdown(floatfmt=".2f")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    results = pd.concat(pd.read_csv(f) for f in args.out.glob("*/results.csv"))
    diagnostics = pd.concat(pd.read_csv(f) for f in args.out.glob("*/diagnostics.csv"))
    df = add_delta(results)

    plot_delta(df, args.out / "delta_vs_baseline.png")

    diag_cols = [
        "actual_noise_rate", "cl_flag_rate", "cl_precision", "cl_recall",
        "naive_trusted_noise", "naive_untrusted_noise",
        "topconf_trusted_noise", "topconf_untrusted_noise",
        "random_trusted_noise", "random_untrusted_noise",
    ]
    diag = diagnostics.groupby(["model", "dataset", "noise", "rate"])[diag_cols].mean() * 100

    lines = ["# Results summary", "", "![](delta_vs_baseline.png)", ""]
    lines += ["## Mean over all conditions (test accuracy %, difference from baseline in pt)", "", fmt(overall_table(df).set_index(["model", "group", "method"])), ""]
    for model in sorted(df["model"].unique()):
        lines += [f"## {model}: Δ by noise kind x noise rate (pt, mean over 3 datasets x 5 seeds)", "",
                  fmt(breakdown_table(df, model, ["noise", "rate"])), ""]
        lines += [f"## {model}: Δ by dataset (pt)", "",
                  fmt(breakdown_table(df, model, ["dataset"])), ""]
    lines += ["## cleanlab detection and noise rates of the pseudo trusted/untrusted sets (%)", "", fmt(diag), ""]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
