#!/usr/bin/env python3
"""
02_latency_summary.py — every closed-loop latency value reported in the paper
==============================================================================
Inputs
  ../data/<session>/6_Latency_<session>.csv      software timestamps per trigger
  ../data/<session>/7_FrameLedger_<session>.csv  capture timestamp of every written frame
  ../data/<session>/8_LatencyMeta_<session>.json session metadata (rates, queue drops)
  ../results/optical_transitions.csv             from 01_optical_transitions.py

Outputs
  ../results/software_latency_summary.csv   medians / IQR per leg, per session and pooled
  ../results/optical_summary.csv            optical latency, camera-acquisition delay D
                                            (bootstrap CI), paired Wilcoxon test
  ../results/animal_referenced_latency.csv  boundary-crossing -> visible-cue latency
  ../results/integrity_checks.csv           data-integrity checks
  ../results/chatter_lat1_1.csv             boundary-chatter statistics (session lat1_1)
  ../results/latency_report.txt             human-readable report of all of the above

Run from this directory:  python3 02_latency_summary.py
Dependencies: numpy, pandas, scipy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RES = HERE.parent / "results"
SESSIONS = ("lat1_1", "lat1_2")
SEED = 42
N_BOOT = 20_000
N_SIM = 200_000
LEGS = {"frame_to_ack_ms": "end-to-end (capture -> acknowledgment)",
        "capture_to_decision_ms": "capture -> decision",
        "decision_to_send_ms": "decision -> serial send",
        "serial_act_ms": "serial transmission + actuation"}

report: list[str] = []


def say(line: str = "") -> None:
    print(line)
    report.append(line)


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except OSError as exc:
        sys.exit(f"Cannot read {path}: {exc}")


def load():
    ev, led, meta = [], {}, {}
    for s in SESSIONS:
        e = read_csv(DATA / s / f"6_Latency_{s}.csv"); e["session"] = s; ev.append(e)
        led[s] = read_csv(DATA / s / f"7_FrameLedger_{s}.csv")
        try:
            meta[s] = json.loads((DATA / s / f"8_LatencyMeta_{s}.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"Cannot read metadata of {s}: {exc}")
    return pd.concat(ev, ignore_index=True), led, meta


def q(x, p):
    return float(np.percentile(x, p))


def software(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, d in [("pooled", ev)] + [(s, g) for s, g in ev.groupby("session")]:
        for col in LEGS:
            x = d[col].to_numpy()
            rows.append({"scope": scope, "leg": col, "n": x.size, "median": np.median(x),
                         "q25": q(x, 25), "q75": q(x, 75)})
    out = pd.DataFrame(rows)
    say("=== Software log ===")
    for _, r in out.iterrows():
        say(f"{r.scope:7s} {LEGS[r.leg]:<40s} n={r.n:3d} median {r['median']:.2f} ms "
            f"[IQR {r.q25:.1f}-{r.q75:.1f}]")
    return out


def integrity(ev: pd.DataFrame, led: dict, meta: dict) -> pd.DataFrame:
    rows = []
    c2d = (ev.t_decision_perf - ev.t_capture_perf) * 1000
    d2s = (ev.t_send_perf - ev.t_decision_perf) * 1000
    ser = (ev.t_ack_perf - ev.t_send_perf) * 1000
    tot = (ev.t_ack_perf - ev.t_capture_perf) * 1000
    rows.append(("max |derived - logged| over the four legs (ms)",
                 float(max((c2d - ev.capture_to_decision_ms).abs().max(),
                           (d2s - ev.decision_to_send_ms).abs().max(),
                           (ser - ev.serial_act_ms).abs().max(),
                           (tot - ev.frame_to_ack_ms).abs().max()))))
    rows.append(("max |sum of legs - end-to-end| (ms)",
                 float((ev.capture_to_decision_ms + ev.decision_to_send_ms + ev.serial_act_ms
                        - ev.frame_to_ack_ms).abs().max())))
    rows.append(("triggers with non-empty acknowledgment",
                 f"{int((ev.ack_ok.astype(str) == 'True').sum())}/{len(ev)}"))
    rows.append(("serial legs < 1 ms (sub-millisecond mode)", int((ev.serial_act_ms < 1).sum())))
    rho, p = stats.spearmanr(ev.serial_act_ms, ev.frame_to_ack_ms)
    rows.append(("Spearman rho, serial leg vs end-to-end", f"{rho:.3f} (p={p:.2g})"))
    for s in SESSIONS:
        L = led[s].sort_values("video_write_index")
        dup = float((L.cam_seq.diff() == 0).mean() * 100)
        dur = L.t_capture_perf.iloc[-1] - L.t_capture_perf.iloc[0]
        m = meta[s]
        rows += [
            (f"{s}: frame ledger contiguous (video_write_index steps of 1)",
             bool((L.video_write_index.diff().dropna() == 1).all())),
            (f"{s}: capture loop rate (Hz, metadata)", round(m["fps_measured"], 2)),
            (f"{s}: camera delivery rate (unique cam_seq / s)",
             round(L.cam_seq.nunique() / dur, 2)),
            (f"{s}: duplicate frames in ledger (%)", round(dup, 1)),
            (f"{s}: video queue drops", m["video_queue_drops"]),
            (f"{s}: analysis queue drops", m["analysis_queue_drops"]),
            (f"{s}: analysis queue drops (% of captures)",
             round(100 * m["analysis_queue_drops"] / m["n_captures_consumed"], 3)),
        ]
    all_drops = sum(meta[s]["analysis_queue_drops"] for s in SESSIONS)
    all_caps = sum(meta[s]["n_captures_consumed"] for s in SESSIONS)
    rows.append(("analysis queue drops, both sessions (% of captures)",
                 round(100 * all_drops / all_caps, 3)))
    out = pd.DataFrame(rows, columns=["check", "value"])
    say("\n=== Integrity checks ===")
    for _, r in out.iterrows():
        say(f"{r.check}: {r.value}")
    return out


def optical(op: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    say("\n=== Optical read-out (isolated transitions) ===")
    for scope, d in [("pooled", op)] + [(s, g) for s, g in op.groupby("session")]:
        D = (d.optical_ms - d.logged_ms).to_numpy()
        row = {"scope": scope, "n": len(d), "optical_median": d.optical_ms.median(),
               "logged_median": d.logged_ms.median(), "D_median": float(np.median(D))}
        if scope == "pooled":
            boot = np.median(rng.choice(D, size=(N_BOOT, D.size), replace=True), axis=1)
            row["D_ci_lo"], row["D_ci_hi"] = q(boot, 2.5), q(boot, 97.5)
            row["wilcoxon_p"] = float(stats.wilcoxon(d.optical_ms, d.logged_ms).pvalue)
        rows.append(row)
        extra = (f" | D 95% CI {row['D_ci_lo']:.1f}-{row['D_ci_hi']:.1f} | paired Wilcoxon "
                 f"p = {row['wilcoxon_p']:.2g}") if scope == "pooled" else ""
        say(f"{scope:7s} n={row['n']:2d} decision->LED median {row['optical_median']:.1f} ms | "
            f"logged {row['logged_median']:.1f} | D {row['D_median']:.1f}{extra}")
    return pd.DataFrame(rows)


def animal_referenced(op: pd.DataFrame, meta: dict, rng) -> pd.DataFrame:
    """Uniform sampling phase over the decision cadence + resampled optical latency."""
    rows, pooled = [], []
    say("\n=== Animal-referenced latency (boundary crossing -> visible cue) ===")
    for s in SESSIONS:
        cad = 1000.0 / meta[s]["fps_measured"]
        o = op.loc[op.session == s, "optical_ms"].to_numpy()
        sim = rng.uniform(0, cad, N_SIM) + rng.choice(o, N_SIM, replace=True)
        pooled.append(sim)
        rows.append({"scope": s, "cadence_ms": cad, "sampling_phase_median": cad / 2,
                     "optical_median": float(np.median(o)), "total_median": float(np.median(sim)),
                     "total_p95": q(sim, 95)})
    sim = np.concatenate(pooled)
    rows.append({"scope": "pooled", "cadence_ms": np.mean([r["cadence_ms"] for r in rows]),
                 "sampling_phase_median": np.mean([r["sampling_phase_median"] for r in rows]),
                 "optical_median": float(op.optical_ms.median()),
                 "total_median": float(np.median(sim)), "total_p95": q(sim, 95)})
    out = pd.DataFrame(rows)
    for _, r in out.iterrows():
        say(f"{r.scope:7s} cadence {r.cadence_ms:.1f} ms | sampling phase {r.sampling_phase_median:.1f} "
            f"| optical {r.optical_median:.1f} | total median {r.total_median:.1f} ms "
            f"(p95 {r.total_p95:.1f})")
    return out


def chatter(ev: pd.DataFrame) -> pd.DataFrame:
    d = ev[ev.session == "lat1_1"].sort_values("t_capture_perf")
    iti = np.diff(d.t_capture_perf.to_numpy()) * 1000
    per_roi = d.roi.value_counts()
    dur_min = (d.t_capture_perf.iloc[-1] - d.t_capture_perf.iloc[0]) / 60
    rows = [("triggers", len(d)), ("span of triggers (min)", round(dur_min, 1)),
            ("median inter-trigger interval (ms)", round(float(np.median(iti)), 1)),
            ("consecutive triggers < 200 ms apart (%)", round(float(np.mean(iti < 200) * 100), 1)),
            ("triggers on the most frequent zone", f"{int(per_roi.iloc[0])} (zone {per_roi.index[0]})")]
    out = pd.DataFrame(rows, columns=["statistic", "value"])
    say("\n=== Boundary chatter, session lat1_1 ===")
    for _, r in out.iterrows():
        say(f"{r.statistic}: {r.value}")
    return out


def main() -> None:
    RES.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    ev, led, meta = load()
    op = read_csv(RES / "optical_transitions.csv")
    software(ev).to_csv(RES / "software_latency_summary.csv", index=False, float_format="%.4f")
    integrity(ev, led, meta).to_csv(RES / "integrity_checks.csv", index=False)
    optical(op, rng).to_csv(RES / "optical_summary.csv", index=False, float_format="%.6g")
    animal_referenced(op, meta, rng).to_csv(RES / "animal_referenced_latency.csv", index=False,
                                            float_format="%.4f")
    chatter(ev).to_csv(RES / "chatter_lat1_1.csv", index=False)
    (RES / "latency_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
