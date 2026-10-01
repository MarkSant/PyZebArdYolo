#!/usr/bin/env python3
"""
01_optical_transitions.py — optical read-out of the closed-loop cue (PyZebArdYolo)
===================================================================================
Measures, from the recorded session video, when each LED actually changed state,
and pairs every isolated logged trigger with the first isolated LED transition of
the right polarity. This captures the full path from frame capture to visible cue,
including the camera-acquisition delay that a software log cannot see.

Pipeline, per session:
  1. Expected ON/OFF state of each zone is reconstructed from the event log
     (6_Latency_<session>.csv; enter -> ON, exit -> OFF).
  2. Matched filter: the green-channel intensity series of every pixel (frames
     downsampled to 320 x 180) is correlated with that state; the pixel with the
     highest correlation locates the LED of each zone (no manual annotation).
  3. Full-rate series: mean green intensity of a 13 x 13 px patch at each LED.
  4. Schmitt trigger (thresholds at 30% and 70% of the 2nd-98th percentile range)
     converts the series into ON/OFF states and transitions.
  5. Zones whose LED contrast (98th - 2nd percentile) is below MIN_CONTRAST gray
     levels are excluded.
  6. Isolation filter: a logged event is used only if no other event on the same
     zone occurs within +/- ISO_S seconds, and it is paired with the first isolated
     transition of matching polarity in [t - 20 ms, t + 1 s].
  7. Video frame index -> time uses t_capture_perf from the frame ledger
     (7_FrameLedger_<session>.csv), so no nominal frame rate is assumed.

Inputs : ../data/<session>/{6_Latency_*.csv, 7_FrameLedger_*.csv, <session>.mp4}
         (the MP4 files, ~0.4 GB each, are archived separately; see ../README.md)
Outputs: ../results/optical_transitions.csv    one row per paired isolated event
         ../results/led_positions_contrast.csv  LED location, correlation, contrast,
                                                 number of detected transitions

Run from this directory:  python3 01_optical_transitions.py
(if the videos are kept outside ../data, set PYZEB_VIDEO_DIR=<folder containing
 lat1_1/lat1_1.mp4 and lat1_2/lat1_2.mp4>)
Dependencies: numpy, pandas, opencv-python
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RES = HERE.parent / "results"
SESSIONS = ("lat1_1", "lat1_2")
SMALL_W, SMALL_H = 320, 180
PATCH_R = 6               # half-width of the full-resolution LED patch (px)
MIN_CONTRAST = 70.0       # gray levels (98th - 2nd percentile)
ISO_S = 1.0               # isolation window (s)
MAX_OPTICAL_MS = 900.0


def load_session(sess: str):
    folder = DATA / sess
    try:
        ev = pd.read_csv(folder / f"6_Latency_{sess}.csv")
        led = pd.read_csv(folder / f"7_FrameLedger_{sess}.csv")
    except OSError as exc:
        sys.exit(f"Cannot read logs of {sess}: {exc}")
    video = folder / f"{sess}.mp4"
    alt = os.environ.get("PYZEB_VIDEO_DIR")       # optional: videos stored elsewhere
    if not video.exists() and alt:
        video = Path(alt) / sess / f"{sess}.mp4"
    if not video.exists():
        sys.exit(f"Missing video {video}. Download the session videos (see ../README.md).")
    ev["key"] = ev["roi"].astype(str)
    led = led.sort_values("video_write_index").reset_index(drop=True)
    return ev.sort_values("t_capture_perf").reset_index(drop=True), led, video


def expected_state(ev: pd.DataFrame, keys, t_frame: np.ndarray, n: int) -> np.ndarray:
    """Step function (ON after 'enter', OFF after 'exit') per zone, per video frame."""
    S = np.zeros((len(keys), n), np.float32)
    for j, k in enumerate(keys):
        st, prev = 0.0, 0
        for _, r in ev[ev["key"] == k].iterrows():
            f = int(np.clip(np.searchsorted(t_frame, r.t_capture_perf), 0, n - 1))
            S[j, prev:f] = st
            st = 1.0 if r.edge == "enter" else 0.0
            prev = f
        S[j, prev:] = st
    return S


def locate_leds(video: Path, S: np.ndarray, keys, n: int) -> dict:
    """Matched-filter localisation (streaming, O(pixels) memory)."""
    cap = cv2.VideoCapture(str(video))
    sx = np.zeros((SMALL_H, SMALL_W)); sxx = np.zeros_like(sx)
    sxs = np.zeros((len(keys), SMALL_H, SMALL_W))
    i = 0
    while i < n:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.resize(fr, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)[:, :, 1].astype(float)
        sx += g; sxx += g * g
        for j in range(len(keys)):
            if S[j, i]:
                sxs[j] += g
        i += 1
    cap.release()
    n = i
    mx = sx / n; vx = sxx / n - mx ** 2
    scale = 1280 / SMALL_W
    out = {}
    for j, k in enumerate(keys):
        ms = S[j, :n].mean(); vs = (S[j, :n] ** 2).mean() - ms ** 2
        rho = (sxs[j] / n - mx * ms) / np.sqrt(np.maximum(vx, 1e-9) * max(vs, 1e-9))
        y, x = np.unravel_index(np.nanargmax(rho), rho.shape)
        out[k] = {"rho": float(rho[y, x]), "y": int(y * scale), "x": int(x * scale)}
    return out, n


def led_series(video: Path, locs: dict, n: int) -> dict:
    cap = cv2.VideoCapture(str(video))
    ser = {k: np.zeros(n) for k in locs}
    i = 0
    while i < n:
        ok, fr = cap.read()
        if not ok:
            break
        for k, v in locs.items():
            y, x = v["y"], v["x"]
            ser[k][i] = fr[max(0, y - PATCH_R):y + PATCH_R + 1,
                           max(0, x - PATCH_R):x + PATCH_R + 1, 1].mean()
        i += 1
    cap.release()
    return {k: v[:i] for k, v in ser.items()}


def schmitt(v: np.ndarray):
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    thi, tlo = lo + 0.7 * (hi - lo), lo + 0.3 * (hi - lo)
    st = np.zeros(v.size, np.int8); cur = 0
    for i, x in enumerate(v):
        if cur == 0 and x > thi:
            cur = 1
        elif cur == 1 and x < tlo:
            cur = 0
        st[i] = cur
    tr = np.flatnonzero(np.diff(st) != 0) + 1
    return tr, st[tr], hi - lo


def isolated(t: np.ndarray) -> np.ndarray:
    gp = np.r_[np.inf, np.diff(t)]; gn = np.r_[np.diff(t), np.inf]
    return (gp > ISO_S) & (gn > ISO_S)


def main() -> None:
    RES.mkdir(exist_ok=True)
    rows, info = [], []
    for sess in SESSIONS:
        ev, led, video = load_session(sess)
        T = led.t_capture_perf.to_numpy()
        keys = sorted(ev["key"].unique())
        S = expected_state(ev, keys, T, len(T))
        locs, n = locate_leds(video, S, keys, len(T))
        ser = led_series(video, locs, n)
        for k in keys:
            tr, pol, contrast = schmitt(ser[k])
            used = contrast >= MIN_CONTRAST
            info.append({"session": sess, "zone": k, **locs[k], "contrast": contrast,
                         "n_transitions": int(tr.size), "n_events": int((ev.key == k).sum()),
                         "used": used})
            print(f"[{sess}] zone {k}: rho={locs[k]['rho']:.3f} contrast={contrast:.0f} "
                  f"transitions={tr.size} events={(ev.key == k).sum()} used={used}")
            if not used:
                continue
            t_tr = T[np.minimum(tr, T.size - 1)]
            iso_tr = isolated(t_tr)
            sub = ev[ev.key == k].reset_index(drop=True)
            iso_ev = isolated(sub.t_capture_perf.to_numpy())
            for i, e in sub.iterrows():
                if not iso_ev[i]:
                    continue
                want = 1 if e.edge == "enter" else 0
                sel = np.flatnonzero((pol == want) & iso_tr &
                                     (t_tr >= e.t_capture_perf - 0.02) &
                                     (t_tr <= e.t_capture_perf + 1.0))
                if sel.size == 0:
                    continue
                opt = (t_tr[sel[0]] - e.t_capture_perf) * 1000.0
                if 0 < opt < MAX_OPTICAL_MS:
                    rows.append({"session": sess, "zone": k, "edge": e.edge,
                                 "event_id": int(e.event_id), "optical_ms": opt,
                                 "logged_ms": float(e.frame_to_ack_ms),
                                 "c2d_ms": float(e.capture_to_decision_ms),
                                 "serial_ms": float(e.serial_act_ms)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "optical_transitions.csv", index=False, float_format="%.4f")
    pd.DataFrame(info).to_csv(RES / "led_positions_contrast.csv", index=False,
                              float_format="%.4f")
    d = out.optical_ms - out.logged_ms
    print(f"paired isolated events: {len(out)} | optical median {out.optical_ms.median():.1f} ms "
          f"| D median {d.median():.1f} ms")


if __name__ == "__main__":
    main()
