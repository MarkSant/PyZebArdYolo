#!/usr/bin/env python3
"""
05_sensitivity_checks.py — supplementary values reported in the paper
======================================================================
1. Pairing-offset sensitivity (Sections 2.6 and 3.2): agreement of each method
   recomputed for pairs at most 4 frames apart and for pairs 5-15 frames apart.
2. Decision cadence of the validation tracks (Section 2.4): median interval between
   consecutive positions in the live-recorded 3_CoordMovimento files.
3. Expected boundary-to-cue latency for that cadence (Section 3.4): uniform sampling
   phase over the cadence + optical latency resampled from
   latency/results/optical_transitions.csv (2 x 10^5 draws, seed 42). An estimate,
   not a measurement.
4. Operator time for manual annotation (Section 3.3), from the timing records of the
   validation study (first annotator: 5 sessions; second annotator: 3 sessions).

Outputs: results/pairing_sensitivity.csv and results/sensitivity_report.txt
Run from this directory:  python3 05_sensitivity_checks.py
Dependencies: numpy, pandas, scipy (via agreement_stats.py)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agreement_stats import icc_a1

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAIRS = HERE / "paired_coords"
RES = HERE / "results"
TRACKS = HERE.parent / "data" / "PyZebArdYolo_tracks"
OPTICAL = REPO / "latency" / "results" / "optical_transitions.csv"
MAX_GAP_CLOSE = 4
SEED = 42
N_SIM = 200_000

# Manual annotation time per 5-min video (minutes), from the study's timing records
TIME_FIRST_ANNOTATOR = np.array([29, 24, 22, 25, 24])   # SEST_Dia1, CEST_Dia1, SEST_Dia2, CEST_Dia2, SEST_Dia3
TIME_SECOND_ANNOTATOR = np.array([43, 35, 28])          # CEST_Dia3, SEST_Dia4, CEST_Dia7

report: list[str] = []


def say(line: str = "") -> None:
    print(line)
    report.append(line)


def read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except OSError as exc:
        sys.exit(f"Cannot read {path}: {exc}")


def pairing_sensitivity() -> pd.DataFrame:
    rows = []
    say("=== 1. Pairing-offset sensitivity ===")
    for m in ("PyZebArdYolo", "ZebTrack", "Observer_2"):
        p = read(PAIRS / f"pares_{m}.csv").dropna(subset=["radial"])
        say(f"{m}: mean offset {p.gap_frames.mean():.2f} frames")
        subsets = (("all", p), (f"<={MAX_GAP_CLOSE}", p[p.gap_frames <= MAX_GAP_CLOSE]),
                   (f">={MAX_GAP_CLOSE + 1}", p[p.gap_frames > MAX_GAP_CLOSE]))
        for lab, s in subsets:
            if len(s) < 3:
                continue
            ix = icc_a1(s[["x_man", "x_met"]].to_numpy())["icc"]
            iy = icc_a1(s[["y_man", "y_met"]].to_numpy())["icc"]
            row = {"metodo": m, "subset": lab, "n": len(s), "pct": 100 * len(s) / len(p),
                   "radial_med": s.radial.median(), "radial_p95": np.percentile(s.radial, 95),
                   "icc_x": ix, "icc_y": iy}
            rows.append(row)
            say(f"   {lab:4s} n={row['n']:4d} ({row['pct']:.1f}%) radial median "
                f"{row['radial_med']:.2f} p95 {row['radial_p95']:.1f} ICC X {ix:.4f} Y {iy:.4f}")
    return pd.DataFrame(rows)


def cadence_ms() -> float:
    files = glob.glob(str(TRACKS / "*" / "*" / "3_CoordMovimento_*.csv"))
    if not files:
        sys.exit(f"No validation tracks found under {TRACKS}")
    dts = [np.median(np.diff(read(Path(f)).timestamp.to_numpy())) * 1000 for f in files]
    say(f"\n=== 2. Decision cadence of the validation tracks: median {np.median(dts):.1f} ms "
        f"(range {min(dts):.1f}-{max(dts):.1f}; n = {len(dts)} sessions)")
    return float(np.median(dts))


def latency_estimate(cad: float) -> None:
    rng = np.random.default_rng(SEED)
    opt = read(OPTICAL).optical_ms.to_numpy()
    sim = rng.uniform(0, cad, N_SIM) + rng.choice(opt, N_SIM, replace=True)
    say(f"\n=== 3. Expected boundary-to-cue latency at a {cad:.1f}-ms cadence: median "
        f"{np.median(sim):.1f} ms, p95 {np.percentile(sim, 95):.1f} ms (estimate)")


def operator_time() -> None:
    a, b = TIME_FIRST_ANNOTATOR, TIME_SECOND_ANNOTATOR
    say(f"\n=== 4. Manual annotation time per 5-min video: first annotator {a.mean():.1f} ± "
        f"{a.std(ddof=1):.1f} min (n = {a.size}); second annotator {b.mean():.1f} ± "
        f"{b.std(ddof=1):.1f} min (n = {b.size})")


def main() -> None:
    RES.mkdir(exist_ok=True)
    pairing_sensitivity().to_csv(RES / "pairing_sensitivity.csv", index=False, float_format="%.4f")
    latency_estimate(cadence_ms())
    operator_time()
    (RES / "sensitivity_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
