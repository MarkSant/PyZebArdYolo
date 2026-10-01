# Changelog

## 1.3.0 — 2026-10-01

Release accompanying the submission of the PyZebArdYolo paper (Journal of
Neuroscience Methods). No change to the acquisition software or firmware.

### Added
- `latency/`: logs of the two closed-loop latency sessions (lat1_1, lat1_2), the
  optical read-out script, the summary script that produces every latency value of
  the paper, integrity checks and Figure 9 (vector PDF).
- `validation/analysis/agreement_stats.py`: single implementation of ICC(A,1) with
  the exact McGraw & Wong 95% CI, Bland–Altman and proportional-bias statistics.
- `validation/analysis/03_agreement_summary.py`: pooled Table 1, Table 2 diagnostics
  (% within LoA, proportional-bias slope, Breusch–Pagan), % of frames within 20/30 px,
  metric conversion and recall summary.
- `validation/analysis/04_figures.py`: Figures 6–8 as vector PDF and 600-dpi PNG.
- `LICENSES/`: full texts of MIT, AGPL-3.0, CERN-OHL-S-2.0 and CC-BY-4.0.

### Changed
- `01_pairing_metrics.py` uses `agreement_stats.py` (exact CIs at full precision
  instead of pingouin's two-decimal CIs); NaN rows no longer propagate into ICCs.
- Results regenerated: `summary_pooled.csv` (exact CIs), `metrics_per_video.csv`
  (CEST_Dia5 no longer NaN), `recall.csv`, `icc_per_video_*.csv`, `mixed_model.txt`
  and the paired coordinates, all regenerated from the current annotation files
  (the previous paired files of CEST_Dia5 contained one empty pair from an empty
  manual row; N is now 4114 for PyZebArdYolo and 4200 for ZebTrack).
- READMEs, `CITATION.cff` (new paper title, authors, journal, concept DOI) and
  version metadata updated.
