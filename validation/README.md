# PyZebArdYolo — validation data & analysis

This folder holds the data and analysis that support the **tracking-fidelity
validation** reported in the PyZebArdYolo paper. Every number, table and figure
of the tracking validation (Tables 1–3, Figures 6–8) is reproducible from the
files here.

**Scope:** **PyZebArdYolo** (the apparatus' own tracking), **raw ZebTrack**
(the comparator, automatic mode) and a **second observer** (inter-observer human
floor), each compared against **manual frame-by-frame annotation** (gold standard).

## Method / folder mapping

| Name in the paper | Folder / label here | What it is |
|---|---|---|
| Manual (gold standard) | `data/Manual_GroundTruth/` | human annotation, from scratch |
| PyZebArdYolo | `data/PyZebArdYolo_tracks/` | the apparatus' YOLO tracking (bbox → centroid) |
| ZebTrack (raw) | `data/ZebTrack_raw/` | ZebTrack in automatic mode, no human correction |
| Observer 2 | `data/Observer_2/` | second annotator (3 videos: CEST_Dia3, SEST_Dia4, CEST_Dia7) |

Sessions: 14 videos, `CEST_Dia1–7` (stress-exposed animal) and `SEST_Dia1–7`
(unstressed control animal), one session per animal and day.

## Directory structure

```
validation/
├── data/                          raw per-session coordinate files
│   ├── Manual_GroundTruth/
│   ├── PyZebArdYolo_tracks/
│   ├── ZebTrack_raw/
│   └── Observer_2/
└── analysis/
    ├── agreement_stats.py         shared ICC(A,1) + exact 95% CI, Bland–Altman, proportional bias
    ├── 01_pairing_metrics.py      frame pairing, recall, per-video ICC / Bland–Altman (Table 3)
    ├── 02_mixed_model.R           random-intercept models, per-video ICC (psych), method comparison
    ├── 03_agreement_summary.py    pooled Table 1, Table 2 diagnostics, % within 20/30 px, mm, recall
    ├── 04_figures.py              Figures 6, 7 and 8 (vector PDF + 600-dpi PNG)
    ├── paired_coords/             frame-paired coordinates per method (pares_*.csv)
    ├── results/                   metrics_per_video, recall, recall_summary, summary_pooled,
    │                              bland_altman_diagnostics, localization_summary,
    │                              icc_per_video_*, mixed_model.txt
    └── figures/                   Fig6–Fig8 (pdf, png) and Bland–Altman plots per method
```

## Conventions (fixed)

- 30 fps; annotation and tracking sampled every 30 frames; image 1280 × 720 px.
- All coordinates are in **image convention** (origin top-left, y down). The
  manual annotation is already converted (`y = 720 − y_matlab`); PyZebArdYolo
  and ZebTrack are already in image convention — **do not flip Y**.
- PyZebArdYolo centroid = bounding-box centre, `((x1+x2)/2, (y1+y2)/2)`.
- Pairing: nearest frame within **≤ 15 frames**; unpaired reference frames are
  counted as recall misses. **Recall** and **localization** are measured separately.
- Agreement: ICC(2,1) absolute agreement with the exact McGraw & Wong 95% CI;
  Bland–Altman bias and 95% limits of agreement per axis; proportional bias
  (difference regressed on mean) and Breusch–Pagan test; a linear mixed model with
  **video as a random intercept** avoids pseudo-replication.
- Imaging scale: 1.24 px mm⁻¹ (560 mm arena length).

## Reproduce

```bash
cd analysis
python3 01_pairing_metrics.py      # numpy, pandas, scipy, statsmodels, matplotlib
Rscript 02_mixed_model.R           # R packages: lme4, psych
python3 03_agreement_summary.py
python3 04_figures.py
```

## Key results (pooled, 14 sessions)

| Method | N pairs | Recall | ICC X [95% CI] | ICC Y [95% CI] | Median radial (px) |
|---|---|---|---|---|---|
| PyZebArdYolo | 4114 | 97.9% | 0.998 [0.998–0.998] | 0.987 [0.986–0.988] | 9.9 |
| ZebTrack (raw) | 4200 | 99.9% | 0.957 [0.954–0.960] | 0.855 [0.846–0.863] | 43.9 |
| Observer 2 (human floor, 3 videos) | 903 | 100% | 0.998 [0.998–0.999] | 0.996 [0.995–0.996] | 5.8 |

Mixed model (radial error, video as random intercept, reference = PyZebArdYolo):
**β(ZebTrack) = +38.0 px** (95% CI 36.5–39.6; t = 47.6).

## License & citation

Data and annotations: **CC BY 4.0**. Analysis scripts: MIT (see the top-level
`LICENSE`, `LICENSES/` and `NOTICE`). If you use these data, please cite the
PyZebArdYolo paper and this repository (`CITATION.cff`).
