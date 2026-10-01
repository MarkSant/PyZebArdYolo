#!/usr/bin/env python3
"""
03_agreement_summary.py — pooled agreement statistics reported in the paper
============================================================================
Reads the frame-paired coordinates written by 01_pairing_metrics.py and
produces every pooled value of the validation section:

    results/summary_pooled.csv           Table 1: N, ICC(2,1) X/Y with exact 95% CI,
                                          Bland-Altman bias and LoA, radial median/p95
    results/bland_altman_diagnostics.csv Table 2: bias [LoA], % within LoA,
                                          proportional-bias slope [95% CI], Breusch-Pagan p,
                                          change in predicted bias across the arena
    results/localization_summary.csv     % of frames within 20 and 30 px, radial median in mm
    results/recall_summary.csv           mean / min / max detection recall per method

Imaging scale: 1.24 px/mm (560 mm tank length imaged at 1280 x 720 px).

Run from this directory:  python3 03_agreement_summary.py
Dependencies: numpy, pandas, scipy, statsmodels (see agreement_stats.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agreement_stats import bland_altman, icc_a1, proportional_bias

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "paired_coords"
RES = HERE / "results"
METHODS = ("PyZebArdYolo", "ZebTrack", "Observer_2")
PX_PER_MM = 1.24
THRESHOLDS_PX = (20, 30)


def load_pairs(method: str) -> pd.DataFrame:
    """Load pares_<method>.csv; exits with a clear message if missing."""
    path = PAIRS / f"pares_{method}.csv"
    try:
        return pd.read_csv(path)
    except OSError as exc:
        sys.exit(f"Cannot read {path}: {exc}. Run 01_pairing_metrics.py first.")


def main() -> None:
    RES.mkdir(exist_ok=True)
    pooled, diag, loc = [], [], []
    for m in METHODS:
        df = load_pairs(m)
        n_all = len(df)
        ok = df.dropna(subset=["x_man", "y_man", "x_met", "y_met", "radial"])
        row = {"metodo": m, "n_pares": n_all, "n_used": len(ok),
               "n_videos": df["video_id"].nunique()}
        for ax in ("x", "y"):
            icc = icc_a1(ok[[f"{ax}_man", f"{ax}_met"]].to_numpy())
            d = (ok[f"{ax}_met"] - ok[f"{ax}_man"]).to_numpy()
            mean = ((ok[f"{ax}_met"] + ok[f"{ax}_man"]) / 2).to_numpy()
            ba = bland_altman(d)
            pb = proportional_bias(mean, d)
            row.update({f"icc_{ax}": icc["icc"], f"icc_{ax}_lo": icc["ci_lo"],
                        f"icc_{ax}_hi": icc["ci_hi"], f"bias_{ax}": ba["bias"],
                        f"loa_{ax}_lo": ba["loa_lo"], f"loa_{ax}_hi": ba["loa_hi"]})
            diag.append({"metodo": m, "axis": ax.upper(), "n": ba["n"], "bias": ba["bias"],
                         "loa_lo": ba["loa_lo"], "loa_hi": ba["loa_hi"],
                         "pct_within_loa": ba["pct_within_loa"], **pb})
        rad = ok["radial"].to_numpy()
        row.update(radial_med=float(np.median(rad)), radial_p95=float(np.percentile(rad, 95)))
        pooled.append(row)
        loc.append({"metodo": m, "radial_med_px": row["radial_med"],
                    "radial_med_mm": row["radial_med"] / PX_PER_MM,
                    **{f"pct_within_{t}px": float(np.mean(rad <= t) * 100) for t in THRESHOLDS_PX},
                    **{f"threshold_{t}px_in_mm": t / PX_PER_MM for t in THRESHOLDS_PX}})

    pooled = pd.DataFrame(pooled)
    diag = pd.DataFrame(diag)
    loc = pd.DataFrame(loc)
    pooled.to_csv(RES / "summary_pooled.csv", index=False, float_format="%.6f")
    diag.to_csv(RES / "bland_altman_diagnostics.csv", index=False, float_format="%.6f")
    loc.to_csv(RES / "localization_summary.csv", index=False, float_format="%.4f")

    try:
        rec = pd.read_csv(RES / "recall.csv")
    except OSError as exc:
        sys.exit(f"Cannot read recall.csv: {exc}. Run 01_pairing_metrics.py first.")
    recs = rec.groupby("metodo").agg(n_videos=("video_id", "count"),
                                     recall_mean=("recall_pct", "mean"),
                                     recall_min=("recall_pct", "min"),
                                     recall_max=("recall_pct", "max")).reset_index()
    recs.to_csv(RES / "recall_summary.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    print("=== Table 1 (pooled) ===")
    for _, r in pooled.iterrows():
        print(f"{r.metodo:<13} N={r.n_pares:5d} (used {r.n_used}) | "
              f"ICC_X {r.icc_x:.3f} [{r.icc_x_lo:.3f}-{r.icc_x_hi:.3f}] | "
              f"ICC_Y {r.icc_y:.3f} [{r.icc_y_lo:.3f}-{r.icc_y_hi:.3f}] | "
              f"radial med {r.radial_med:.2f} p95 {r.radial_p95:.1f}")
    print("\n=== Table 2 (Bland-Altman diagnostics) ===")
    for _, r in diag.iterrows():
        print(f"{r.metodo:<13} {r.axis} bias {r.bias:+.2f} [{r.loa_lo:+.1f}, {r.loa_hi:+.1f}] | "
              f"in LoA {r.pct_within_loa:.1f}% | slope {r.slope:+.4f} "
              f"[{r.slope_ci_lo:+.4f}, {r.slope_ci_hi:+.4f}] | BP p {r.bp_p:.3f} | "
              f"bias change across arena {r.bias_change_across_range:.1f} px")
    print("\n=== Localization thresholds ===")
    print(loc.round(2).to_string(index=False))
    print("\n=== Recall ===")
    print(recs.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
