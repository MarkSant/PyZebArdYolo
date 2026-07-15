#!/usr/bin/env python3
"""
01_pairing_metrics.py — PyZebArdYolo validation (frame pairing + agreement metrics)
===================================================================================
Compares each automated method against the manual annotation (gold standard),
frame by frame, and computes detection recall (separately) and localisation
agreement (ICC(2,1) absolute-agreement, Bland-Altman, radial error).

Methods (P1 scope):
    PyZebArdYolo  — the apparatus' own tracking script (bbox -> centroid)
    ZebTrack      — raw ZebTrack (automatic mode, no human correction)
    Observer_2    — a second human annotator (inter-observer / human floor)

Conventions (fixed):
    - 30 fps; annotation/tracking every 30 frames; image 1280x720
    - Manual already in image convention (y = 720 - y_matlab)
    - PyZebArdYolo / ZebTrack already in image convention (Y down) -> do NOT flip
    - PyZebArdYolo centroid = (x1+x2)/2, (y1+y2)/2
    - Pairing: nearest frame within <= 15 frames; unpaired frames = recall miss
    - Recall (missing frames) and localisation (ICC/BA) are computed SEPARATELY

Note: the DRerio LogAI tracker is a separate platform, validated in its own
repository; it is intentionally not part of this deposit.

Inputs : ../data/{Manual_GroundTruth, PyZebArdYolo_tracks, ZebTrack_raw, Observer_2}
Outputs: paired_coords/  results/  figures/   (next to this script)

Dependencies: pandas, numpy, pingouin (>=0.5), matplotlib
Author: Marco Antonio Sant'Ana Camargos - FAPESP 2023/14200-3
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import pingouin as pg
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "pingouin", "-q"], check=True)
    import pingouin as pg

# ── Paths (relative to this script) ─────────────────────────────────────────
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "data"
for d in ("paired_coords", "results", "figures"):
    (HERE / d).mkdir(parents=True, exist_ok=True)

VIDEOS = (
    [{"id": f"CEST_Dia{d}", "grupo": "CEST", "dia": d, "dia_manual": f"Dia_{d}"} for d in range(1, 8)]
  + [{"id": f"SEST_Dia{d}", "grupo": "SEST", "dia": d, "dia_manual": f"Dia_{d}"} for d in range(1, 8)]
)

TOL_FRAMES = 15   # pairing tolerance (frames)
FLAG_PX    = 40   # informational flag: PyZebArdYolo median radial above this

# ── Loaders ─────────────────────────────────────────────────────────────────
def load_manual(v):
    nome = f"{v['grupo']}_1"
    path = BASE / "Manual_GroundTruth" / v["dia_manual"] / nome / f"{nome}_csv" / f"{nome}__todas_trilhas.csv"
    df = pd.read_csv(path)
    return df[["frame", "x", "y"]].rename(columns={"x": "x_man", "y": "y_man"})

def load_pyzebardyolo(v):
    """Bounding box -> centroid. Y already in image convention (do not flip)."""
    nome = f"{v['grupo']}_1"
    path = BASE / "PyZebArdYolo_tracks" / v["dia_manual"] / nome / f"3_CoordMovimento_{nome}.csv"
    df = pd.read_csv(path)
    df["x_met"] = (df["x1"] + df["x2"]) / 2.0
    df["y_met"] = (df["y1"] + df["y2"]) / 2.0
    return df[["frame", "x_met", "y_met"]]

def load_zebtrack(v):
    nome = f"{v['grupo']}_1"
    path = BASE / "ZebTrack_raw" / v["dia_manual"] / nome / f"{nome}_csv" / f"{nome}__todas_trilhas.csv"
    df = pd.read_csv(path)
    return df[["frame", "x", "y"]].rename(columns={"x": "x_met", "y": "y_met"})

def load_observer2(v):
    nome = f"{v['grupo']}_1"
    path = BASE / "Observer_2" / v["dia_manual"] / nome / f"{nome}_csv" / f"{nome}__todas_trilhas.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[["frame", "x", "y"]].rename(columns={"x": "x_met", "y": "y_met"})

METHODS = {
    "PyZebArdYolo": load_pyzebardyolo,
    "ZebTrack":     load_zebtrack,
    "Observer_2":   load_observer2,
}

# ── Pairing ─────────────────────────────────────────────────────────────────
def pair(manual, method, tol=TOL_FRAMES):
    mf, mx, my = method["frame"].values, method["x_met"].values, method["y_met"].values
    rows, n_miss = [], 0
    for _, r in manual.iterrows():
        diffs = np.abs(mf - r["frame"]); idx = int(np.argmin(diffs)); gap = diffs[idx]
        if gap > tol:
            n_miss += 1; continue
        dx = mx[idx] - r["x_man"]; dy = my[idx] - r["y_man"]
        rows.append({"frame_man": r["frame"], "frame_met": mf[idx], "gap_frames": gap,
                     "x_man": r["x_man"], "y_man": r["y_man"], "x_met": mx[idx], "y_met": my[idx],
                     "dx": dx, "dy": dy, "radial": float(np.hypot(dx, dy))})
    return pd.DataFrame(rows), n_miss

# ── ICC(2,1) absolute agreement ─────────────────────────────────────────────
def icc21(pairs, axis):
    empty = {"icc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    if len(pairs) < 3:
        return empty
    man = "x_man" if axis == "x" else "y_man"; met = "x_met" if axis == "x" else "y_met"
    n = len(pairs)
    long = pd.DataFrame({"subject": list(range(n)) * 2, "rater": ["manual"] * n + ["method"] * n,
                         "rating": pairs[man].tolist() + pairs[met].tolist()})
    try:
        res = pg.intraclass_corr(data=long, targets="subject", raters="rater", ratings="rating")
        mask = res["Type"].isin(["ICC(A,1)", "ICC2"])
        if not mask.any():
            return empty
        row = res[mask].iloc[0]; ci = row["CI95"] if "CI95" in res.columns else row["CI95%"]
        return {"icc": float(row["ICC"]), "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "p": float(row["pval"])}
    except Exception as e:
        print(f"    [ICC error] {e}"); return empty

# ── Bland-Altman stats ──────────────────────────────────────────────────────
def ba_stats(pairs, axis):
    if axis == "radial":
        d = pairs["radial"]; bias = float(d.mean()); loa = float(1.96 * d.std())
        return {"bias": bias, "loa_lo": max(0.0, bias - loa), "loa_hi": bias + loa,
                "median": float(d.median()), "p95": float(np.percentile(d, 95))}
    man = "x_man" if axis == "x" else "y_man"; met = "x_met" if axis == "x" else "y_met"
    diff = pairs[met] - pairs[man]; bias = float(diff.mean()); loa = float(1.96 * diff.std())
    return {"bias": bias, "loa_lo": bias - loa, "loa_hi": bias + loa}

# ── Main pipeline ───────────────────────────────────────────────────────────
metrics, all_pairs = [], {m: [] for m in METHODS}
print("=" * 72); print("AGREEMENT PIPELINE — PyZebArdYolo validation"); print("=" * 72)

for v in VIDEOS:
    manual = load_manual(v); n_man = len(manual)
    for name, loader in METHODS.items():
        met_df = loader(v)
        if met_df is None:
            continue
        pairs, n_miss = pair(manual, met_df)
        recall = 100.0 * (1 - n_miss / n_man) if n_man else np.nan
        if len(pairs) < 3:
            print(f"  [WARN] {v['id']}/{name}: only {len(pairs)} pairs — skipped."); continue
        ix, iy = icc21(pairs, "x"), icc21(pairs, "y")
        bx, by, br = ba_stats(pairs, "x"), ba_stats(pairs, "y"), ba_stats(pairs, "radial")
        rmed = float(pairs["radial"].median()); rp95 = float(np.percentile(pairs["radial"], 95))
        print(f"  {v['id']:12s} | {name:12s} | n={len(pairs):3d} | recall={recall:5.1f}% | "
              f"ICC_X={ix['icc']:.4f} | ICC_Y={iy['icc']:.4f} | radial_med={rmed:5.1f}px")
        pairs["video_id"], pairs["metodo"], pairs["grupo"], pairs["dia"] = v["id"], name, v["grupo"], v["dia"]
        all_pairs[name].append(pairs.copy())
        metrics.append({"video_id": v["id"], "grupo": v["grupo"], "dia": v["dia"], "metodo": name,
                        "n_manual": n_man, "n_pares": len(pairs), "n_sem_par": n_miss, "recall_pct": recall,
                        "icc_x": ix["icc"], "icc_x_ci_lo": ix["ci_lo"], "icc_x_ci_hi": ix["ci_hi"],
                        "icc_y": iy["icc"], "icc_y_ci_lo": iy["ci_lo"], "icc_y_ci_hi": iy["ci_hi"],
                        "bias_x": bx["bias"], "loa_lo_x": bx["loa_lo"], "loa_hi_x": bx["loa_hi"],
                        "bias_y": by["bias"], "loa_lo_y": by["loa_lo"], "loa_hi_y": by["loa_hi"],
                        "radial_mediana": rmed, "radial_p95": rp95,
                        "bias_radial": br["bias"], "loa_hi_radial": br["loa_hi"]})

# ── Save paired coords + tables ─────────────────────────────────────────────
for name, lst in all_pairs.items():
    if lst:
        df = pd.concat(lst, ignore_index=True)
        df.to_csv(HERE / "paired_coords" / f"pares_{name}.csv", index=False, float_format="%.4f")
dfm = pd.DataFrame(metrics)
dfm.to_csv(HERE / "results" / "metrics_per_video.csv", index=False, float_format="%.4f")
dfm[["video_id", "grupo", "dia", "metodo", "n_manual", "n_pares", "n_sem_par", "recall_pct"]].to_csv(
    HERE / "results" / "recall.csv", index=False, float_format="%.2f")

# ── Informational flag ──────────────────────────────────────────────────────
susp = dfm[(dfm["metodo"] == "PyZebArdYolo") & (dfm["radial_mediana"] > FLAG_PX)]
print("=" * 72)
if len(susp):
    print(f"[CHECK] PyZebArdYolo median radial > {FLAG_PX}px (inspect annotation/video):")
    print(susp[["video_id", "radial_mediana"]].to_string(index=False))
else:
    print(f"[OK] No video with PyZebArdYolo median radial > {FLAG_PX}px.")

# ── Bland-Altman figures ────────────────────────────────────────────────────
CORES = {"CEST": "#2196F3", "SEST": "#F44336"}
def plot_ba(name, df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Bland-Altman — {name} vs manual (gold standard)", fontsize=13, fontweight="bold")
    for ax, ax_name, mc, xc in [(axes[0], "X", "x_man", "x_met"), (axes[1], "Y", "y_man", "y_met")]:
        ad, aa = [], []
        for g in ("CEST", "SEST"):
            s = df[df["grupo"] == g]
            if s.empty: continue
            d = s[xc] - s[mc]; a = (s[mc] + s[xc]) / 2
            ax.scatter(a, d, alpha=0.35, s=5, color=CORES[g], label=g, rasterized=True)
            ad += d.tolist(); aa += a.tolist()
        if not ad: continue
        arr = np.array(ad); bias = arr.mean(); loa = 1.96 * arr.std()
        ax.axhline(bias, color="black", lw=1.6, label=f"bias = {bias:+.1f}")
        ax.axhline(bias + loa, color="#E65100", ls="--", lw=1.2, label=f"+1.96SD = {bias+loa:+.1f}")
        ax.axhline(bias - loa, color="#E65100", ls="--", lw=1.2, label=f"-1.96SD = {bias-loa:+.1f}")
        ax.set_xlabel(f"mean(manual, {name}) [{ax_name}, px]"); ax.set_ylabel(f"{name} - manual [{ax_name}, px]")
        ax.set_title(f"{ax_name}-axis (n={len(arr)})"); ax.legend(fontsize=8); ax.grid(alpha=0.22)
    plt.tight_layout(); fig.savefig(HERE / "figures" / f"BA_{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

for name, lst in all_pairs.items():
    if lst: plot_ba(name, pd.concat(lst, ignore_index=True))

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72); print("SUMMARY BY METHOD (means over videos)"); print("=" * 72)
print(f"{'method':<14}{'recall%':>8}{'ICC_X':>9}{'ICC_Y':>9}{'rad_med':>9}{'rad_p95':>9}{'bias_X':>8}{'bias_Y':>8}")
for m in ("PyZebArdYolo", "ZebTrack", "Observer_2"):
    s = dfm[dfm["metodo"] == m]
    if s.empty: continue
    print(f"{m:<14}{s['recall_pct'].mean():>8.1f}{s['icc_x'].mean():>9.4f}{s['icc_y'].mean():>9.4f}"
          f"{s['radial_mediana'].mean():>9.1f}{s['radial_p95'].mean():>9.1f}{s['bias_x'].mean():>+8.2f}{s['bias_y'].mean():>+8.2f}")
print("\nNext: run 02_mixed_model.R (from this directory) for the mixed-effects models.")
