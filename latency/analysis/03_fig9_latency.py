#!/usr/bin/env python3
"""
03_fig9_latency.py — Figure 9 of the paper (vector PDF + 600-dpi PNG)
======================================================================
(A) distribution of the software-logged end-to-end latency (capture -> acknowledgment)
(B) distribution of the optical decision-to-LED latency, with the camera-acquisition
    delay D = median(optical - logged) in the panel title.

Inputs : ../data/<session>/6_Latency_<session>.csv, ../results/optical_transitions.csv
Outputs: ../figures/Fig9_closed_loop_latency.{pdf,png}
Run from this directory:  python3 03_fig9_latency.py
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
DATA = HERE.parent / "data"
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
MM = 1 / 25.4
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
})


def read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except OSError as exc:
        sys.exit(f"Cannot read {path}: {exc}")


def main() -> None:
    sw = pd.concat([read(DATA / s / f"6_Latency_{s}.csv") for s in ("lat1_1", "lat1_2")])
    op = read(RES / "optical_transitions.csv")
    e2e = sw.frame_to_ack_ms.to_numpy()
    opt = op.optical_ms.to_numpy()
    D = float(np.median(op.optical_ms - op.logged_ms))

    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 65 * MM))
    xmax = np.percentile(e2e, 99.5)
    a.hist(e2e[e2e <= xmax], bins=np.arange(0, xmax + 3, 3), color="#0072B2",
           edgecolor="white", linewidth=0.3)
    a.axvline(np.median(e2e), color="black", ls="--", lw=0.9)
    a.text(np.median(e2e), a.get_ylim()[1] * 0.95, f"  median {np.median(e2e):.1f} ms",
           va="top", fontsize=7)
    a.set(title=f"A  Software log (capture → acknowledgment), n = {e2e.size}",
          xlabel="End-to-end latency (ms)", ylabel="Triggers (n)")

    b.hist(opt, bins=np.arange(np.floor(opt.min() / 6) * 6, opt.max() + 6, 6), color="#E69F00",
           edgecolor="white", linewidth=0.3)
    b.axvline(np.median(opt), color="black", ls="--", lw=0.9)
    b.text(np.median(opt), b.get_ylim()[1] * 0.95, f"  median {np.median(opt):.1f} ms",
           va="top", fontsize=7)
    b.set(title=f"B  Optical (decision → LED), n = {opt.size}; D = {D:.1f} ms",
          xlabel="Decision-to-LED latency (ms)", ylabel="Transitions (n)")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"Fig9_closed_loop_latency.{ext}", bbox_inches="tight", **kw)
    print("written: Fig9_closed_loop_latency (pdf, png)")


if __name__ == "__main__":
    main()
