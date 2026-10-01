# PyZebArdYolo — closed-loop latency data & analysis

Data and scripts behind the closed-loop latency results of the PyZebArdYolo
paper (Section 3.4, Figure 9) and the boundary-chatter limitation (Section 3.5).
Every latency value in the paper is produced by `analysis/02_latency_summary.py`.

## Sessions

Two 10-min live sessions on the reference setup (Logitech C270, Arduino Uno R3,
Samsung Galaxy Book Pro 360, OpenVINO on Iris Xe), recorded with release 1.2.0,
in which the latency instrumentation was corrected:

| Session | Triggers | Files |
|---|---|---|
| `lat1_1` | 261 | `data/lat1_1/` |
| `lat1_2` | 107 | `data/lat1_2/` |

Per session: `1_ProcessingArea`, `2_AreasOfInterest`, `3_CoordMovimento` (as in any
session), `6_Latency` (one row per trigger: capture, decision, send and
acknowledgment timestamps from `time.perf_counter`, plus derived legs),
`7_FrameLedger` (capture timestamp and camera sequence number of every written
frame) and `8_LatencyMeta` (measured rates, queue drops, ROI configuration).

**Videos.** The session videos (`lat1_1.mp4`, 0.38 GB; `lat1_2.mp4`, 0.44 GB) exceed
GitHub's file limit and are archived on Zenodo: [DOI to be added]. They are needed
only by step 1. Place them at `data/<session>/<session>.mp4`, or set
`PYZEB_VIDEO_DIR` to a folder containing `lat1_1/lat1_1.mp4` and `lat1_2/lat1_2.mp4`.

## Reproduce

```bash
cd analysis
python3 01_optical_transitions.py   # needs the videos; numpy, pandas, opencv-python (~2 min)
python3 02_latency_summary.py       # numpy, pandas, scipy
python3 03_fig9_latency.py          # matplotlib
```

Step 1 writes `results/optical_transitions.csv` (already included, so steps 2–3 run
without the videos) and `results/led_positions_contrast.csv`.

## Outputs (`results/`)

| File | Content |
|---|---|
| `software_latency_summary.csv` | medians and IQR of each latency leg, per session and pooled |
| `optical_transitions.csv` | isolated LED transitions paired with logged triggers (n = 36) |
| `led_positions_contrast.csv` | LED location (matched filter), correlation, contrast, transitions |
| `optical_summary.csv` | decision-to-LED latency, camera-acquisition delay D (bootstrap CI), Wilcoxon test |
| `animal_referenced_latency.csv` | boundary-crossing-to-visible-cue latency (simulation, 2 × 10⁵ draws) |
| `integrity_checks.csv` | timestamp consistency, acknowledgments, ledger contiguity, duplicate frames, drops |
| `chatter_lat1_1.csv` | inter-trigger statistics documenting boundary chatter |
| `latency_report.txt` | all of the above in readable form |

Figure 9: `figures/Fig9_closed_loop_latency.{pdf,png}`.

## Key results

Software end-to-end (capture → acknowledgment): median 53.6 ms (IQR 45.5–60.3; n = 368).
Optical decision → LED: median 95.3 ms (n = 36); camera-acquisition delay
D = 36.4 ms (bootstrap 95% CI 31.9–42.8). Boundary crossing → visible cue:
≈ 104 ms (median of the pooled simulation).

## License

Data: CC BY 4.0. Scripts: MIT (see the top-level `LICENSE` and `LICENSES/`).
