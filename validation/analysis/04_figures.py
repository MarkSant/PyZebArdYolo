#!/usr/bin/env python3
"""
04_figures.py — validation figures of the paper (vector PDF + 600-dpi PNG)
===========================================================================
    figures/Fig6_radial_error_by_method.{pdf,png}   radial error, box plot per method
    figures/Fig7_bland_altman_PyZebArdYolo.{pdf,png} Bland-Altman, X and Y
    figures/Fig8_radial_error_per_video.{pdf,png}    median radial error per session

Input: paired_coords/pares_*.csv (from 01_pairing_metrics.py).
Colours: Okabe-Ito (colour-blind safe). Fonts embedded as TrueType (Type 42).
Run from this directory:  python3 04_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "paired_coords"
FIG = HERE / "figures"
MM = 1 / 25.4
COL = {"PyZebArdYolo": "#0072B2", "ZebTrack": "#E69F00", "Observer_2": "#009E73"}
LABEL = {"PyZebArdYolo": "PyZebArdYolo", "ZebTrack": "ZebTrack (raw)",
         "Observer_2": "Inter-observer"}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
})


def load(method: str) -> pd.DataFrame:
    path = PAIRS / f"pares_{method}.csv"
    try:
        df = pd.read_csv(path)
    except OSError as exc:
        sys.exit(f"Cannot read {path}: {exc}. Run 01_pairing_metrics.py first.")
    return df.dropna(subset=["x_man", "y_man", "x_met", "y_met", "radial"])


def save(fig, stem: str) -> None:
    FIG.mkdir(exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("written:", stem)


def fig6(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM))
    order = ["PyZebArdYolo", "ZebTrack", "Observer_2"]
    vals = [data[m]["radial"].to_numpy() for m in order]
    bp = ax.boxplot(vals, whis=1.5, showfliers=False, patch_artist=True, widths=0.55,
                    medianprops={"color": "black", "lw": 1.2})
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(COL[m]); patch.set_alpha(0.85); patch.set_linewidth(0.6)
    for i, v in enumerate(vals, start=1):
        med = np.median(v)
        ax.text(i + 0.32, med, f"{med:.1f} px", va="center", fontsize=7)
    ax.set_xticks([1, 2, 3], [LABEL[m] for m in order])
    ax.set_ylabel("Radial error (px)")
    ax.set_title("Localization error vs. manual gold standard")
    save(fig, "Fig6_radial_error_by_method")


def fig7(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 70 * MM), sharey=False)
    for ax, axis in zip(axes, ("x", "y")):
        mean = (df[f"{axis}_met"] + df[f"{axis}_man"]) / 2
        diff = df[f"{axis}_met"] - df[f"{axis}_man"]
        bias, sd = diff.mean(), diff.std(ddof=1)
        ax.scatter(mean, diff, s=3, alpha=0.25, color=COL["PyZebArdYolo"],
                   linewidths=0, rasterized=False)
        ax.axhline(bias, color="black", lw=1.0, label=f"bias = {bias:+.1f} px")
        for k, lab in ((1, "+1.96 SD"), (-1, "−1.96 SD")):
            y = bias + k * 1.96 * sd
            ax.axhline(y, color="#D55E00", lw=0.9, ls="--", label=f"{lab} = {y:+.1f} px")
        ax.set_xlabel(f"Mean of manual and PyZebArdYolo, {axis.upper()} (px)")
        ax.set_ylabel(f"PyZebArdYolo − manual, {axis.upper()} (px)")
        ax.set_title(f"{axis.upper()} axis (n = {len(diff)})")
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "Fig7_bland_altman_PyZebArdYolo")


def fig8(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(180 * MM, 70 * MM))
    floor = float(np.median(data["Observer_2"]["radial"]))
    vids = sorted(data["PyZebArdYolo"]["video_id"].unique(),
                  key=lambda v: (v.split("_")[0], int(v.split("Dia")[1])))
    x = np.arange(len(vids))
    for m, mk in (("PyZebArdYolo", "o"), ("ZebTrack", "^")):
        med = data[m].groupby("video_id")["radial"].median().reindex(vids)
        ax.plot(x, med.values, mk, color=COL[m], ms=5, label=LABEL[m])
    ax.axhline(floor, color="grey", ls="--", lw=0.9,
               label=f"Inter-observer floor ({floor:.1f} px)")
    ax.set_xticks(x, [v.replace("_Dia", " d") for v in vids], rotation=45, ha="right")
    ax.set_ylabel("Median radial error (px)")
    ax.set_title("Median radial error per session (14 videos)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    top = max(np.nanmax(data[m].groupby("video_id")["radial"].median()) for m in ("PyZebArdYolo", "ZebTrack"))
    ax.set_ylim(0, top * 1.25)   # headroom for the legend
    save(fig, "Fig8_radial_error_per_video")


def main() -> None:
    data = {m: load(m) for m in ("PyZebArdYolo", "ZebTrack", "Observer_2")}
    fig6(data)
    fig7(data["PyZebArdYolo"])
    fig8(data)


if __name__ == "__main__":
    main()
