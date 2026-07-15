# PyZebArdYolo — validation data & analysis

This folder holds the data and analysis that support the **tracking-fidelity
validation** reported in the PyZebArdYolo hardware paper (HardwareX). Every
number, table and figure in the paper's *Validation and characterization*
section is reproducible from the files here.

> **Scope.** This deposit covers the P1 apparatus only: **PyZebArdYolo** (the
> apparatus' own tracking script), **raw ZebTrack** (the incumbent comparator,
> automatic mode) and a **second observer** (inter-observer human floor), each
> compared against **manual frame-by-frame annotation** (gold standard).
> The *DRerio LogAI* platform is a separate product, validated in its own
> repository, and is intentionally **not** included here.

## Method / folder mapping

| Name in the paper | Folder / label here | What it is |
|---|---|---|
| Manual (gold standard) | `data/Manual_GroundTruth/` | human annotation, from scratch, blind |
| PyZebArdYolo (this apparatus) | `data/PyZebArdYolo_tracks/` | the apparatus' YOLO tracking script (bbox → centroid) |
| ZebTrack (raw) | `data/ZebTrack_raw/` | ZebTrack in automatic mode, no human correction |
| Observer 2 | `data/Observer_2/` | second annotator (3 videos: CEST_Dia3, SEST_Dia4, CEST_Dia7) |

## Directory structure

```
validation/
├── data/                         raw per-session coordinate files (frame,tempo_s,x,y,trilha)
│   ├── Manual_GroundTruth/
│   ├── PyZebArdYolo_tracks/
│   ├── ZebTrack_raw/
│   └── Observer_2/
└── analysis/
    ├── 01_pairing_metrics.py     frame pairing + recall + ICC(2,1) + Bland–Altman
    ├── 02_mixed_model.R          random-intercept models + per-video ICC + comparison
    ├── paired_coords/            frame-paired coordinates per method (pares_*.csv)
    ├── results/                  metrics_per_video, recall, summary_pooled, icc_per_video_*, mixed_model.txt
    └── figures/                  Bland–Altman plots per method
```

## Conventions (fixed)

- 30 fps; annotation and tracking sampled every 30 frames; image 1280 × 720 px.
- All coordinates are in **image convention** (origin top-left, y down). The
  manual annotation is already converted (`y = 720 − y_matlab`); PyZebArdYolo
  and ZebTrack are already in image convention — **do not flip Y**.
- PyZebArdYolo centroid = bounding-box centre, `((x1+x2)/2, (y1+y2)/2)`.
- Pairing: nearest frame within **≤ 15 frames**; unpaired reference frames are
  counted as recall misses. **Recall** (missing detections) and **localisation**
  (ICC / Bland–Altman / radial error) are measured **separately**.
- Agreement: ICC(2,1) absolute-agreement with 95% CI; Bland–Altman bias and
  95% limits of agreement per axis; a linear mixed model with **video as a
  random effect** avoids pseudo-replication.

## Reproduce

```bash
cd analysis
python 01_pairing_metrics.py     # pandas, numpy, pingouin>=0.5, matplotlib
Rscript 02_mixed_model.R         # R packages: lme4, psych
```

`01` writes the paired coordinates, per-video metrics, recall table and
Bland–Altman figures; `02` fits the mixed models, writes the per-video ICC
tables and `mixed_model.txt`.

## Key results (pooled, 14 sessions)

| Method | Recall | ICC X | ICC Y | Median radial (px) |
|---|---|---|---|---|
| PyZebArdYolo | 97.9% | 0.998 | 0.987 | 9.9 |
| ZebTrack (raw) | 99.9% | 0.957 | 0.855 | 43.9 |
| Observer 2 (human floor) | 100% | 0.999 | 0.996 | 5.8 |

Mixed model (radial, video as random effect, reference = PyZebArdYolo):
**β(ZebTrack) = +38.0 px** (95% CI 36.5–39.6; t = 47.6) — the raw comparator is
~38 px less accurate than PyZebArdYolo on the same gold standard.

## License & citation

Data and annotations are released under **CC BY 4.0**. The analysis scripts
follow the repository licence (see the top-level `LICENSE` / `NOTICE`). If you
use these data, please cite the PyZebArdYolo paper and this repository
(`CITATION.cff`).
