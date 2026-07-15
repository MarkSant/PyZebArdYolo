# 02_mixed_model.R — PyZebArdYolo validation (mixed-effects models + per-video ICC)
# =============================================================================
# Reads the paired-coordinate CSVs produced by 01_pairing_metrics.py and fits:
#   - per-method random-intercept models: (x/y/radial diff) ~ 1 + (1 | video_id)
#     -> fixed intercept = global bias; random effect = between-video variance
#   - per-video ICC(2,1) absolute agreement via psych::ICC()
#   - method comparison: radial ~ method + (1 | video_id), ref = PyZebArdYolo
#
# P1 scope: PyZebArdYolo, ZebTrack (raw), Observer_2. The DRerio LogAI tracker
# is validated separately in its own repository and is not part of this deposit.
#
# Run from this directory:  Rscript 02_mixed_model.R
# Requires: lme4, psych  (base R for everything else)
# Author: Marco Antonio Sant'Ana Camargos - FAPESP 2023/14200-3
# =============================================================================

suppressPackageStartupMessages({ library(lme4); library(psych) })

PARES <- "paired_coords"
RES   <- "results"
dir.create(RES, showWarnings = FALSE, recursive = TRUE)
methods <- c("PyZebArdYolo", "ZebTrack", "Observer_2")

out <- file(file.path(RES, "mixed_model.txt"), open = "wt")
cat("MIXED-EFFECTS MODELS — PyZebArdYolo validation\n", file = out)
cat(sprintf("Run: %s | %s\n", Sys.time(), R.version$version.string), file = out)

icc_per_video <- function(df, axis = "x") {
  man <- if (axis == "x") "x_man" else "y_man"
  met <- if (axis == "x") "x_met" else "y_met"
  vids <- unique(df$video_id); res <- list()
  for (v in vids) {
    s <- df[df$video_id == v, ]; s <- s[complete.cases(s[, c(man, met)]), ]
    val <- tryCatch({
      r <- psych::ICC(cbind(s[[man]], s[[met]]), lmer = FALSE)$results
      rn <- if ("ICC2" %in% rownames(r)) "ICC2" else rownames(r)[2]
      c(r[rn, "ICC"], r[rn, "lower bound"], r[rn, "upper bound"])
    }, error = function(e) c(NA, NA, NA))
    res[[v]] <- data.frame(video_id = v, n_pares = nrow(s),
                           icc = val[1], icc_lb = val[2], icc_ub = val[3])
  }
  do.call(rbind, res)
}

analyse <- function(m) {
  f <- file.path(PARES, sprintf("pares_%s.csv", m))
  if (!file.exists(f)) { cat(sprintf("\n[skip] missing %s\n", f), file = out); return(NULL) }
  df <- read.csv(f)
  df$x_diff <- df$x_met - df$x_man; df$y_diff <- df$y_met - df$y_man
  cat("\n", strrep("=", 70), "\n", sep = "", file = out)
  cat(sprintf("METHOD: %s  (N pairs = %d, videos = %d)\n", m, nrow(df), length(unique(df$video_id))), file = out)
  cat(strrep("=", 70), "\n", sep = "", file = out)
  if (length(unique(df$video_id)) >= 2) {
    for (resp in c("x_diff", "y_diff", "radial")) {
      cat(sprintf("\n--- %s ~ 1 + (1|video_id) ---\n", resp), file = out)
      tryCatch({
        mod <- lmer(as.formula(sprintf("%s ~ 1 + (1|video_id)", resp)), data = df, REML = TRUE)
        b <- fixef(mod)[1]; se <- sqrt(diag(vcov(mod)))[1]
        vc <- as.data.frame(VarCorr(mod))
        cat(sprintf("  intercept (bias) = %.3f [%.3f, %.3f]\n", b, b - 1.96*se, b + 1.96*se), file = out)
        cat(sprintf("  var between-video = %.4f | var residual = %.4f\n", vc$vcov[1], vc$vcov[2]), file = out)
      }, error = function(e) cat(sprintf("  [err] %s\n", e$message), file = out))
    }
  }
  ix <- icc_per_video(df, "x"); iy <- icc_per_video(df, "y")
  tab <- data.frame(video_id = ix$video_id, n_pares = ix$n_pares,
                    icc_x = ix$icc, icc_x_lb = ix$icc_lb, icc_x_ub = ix$icc_ub,
                    icc_y = iy$icc, icc_y_lb = iy$icc_lb, icc_y_ub = iy$icc_ub)
  agg <- aggregate(cbind(radial_mediana = radial) ~ video_id, df, median)
  tab <- merge(tab, agg, by = "video_id")
  w <- tab$n_pares
  cat(sprintf("\n  weighted ICC_X = %.4f | ICC_Y = %.4f | median radial = %.1f px\n",
              weighted.mean(tab$icc_x, w, na.rm = TRUE), weighted.mean(tab$icc_y, w, na.rm = TRUE),
              weighted.mean(tab$radial_mediana, w, na.rm = TRUE)), file = out)
  write.csv(tab, file.path(RES, sprintf("icc_per_video_%s.csv", m)), row.names = FALSE)
}

for (m in methods) analyse(m)

# ── Method comparison: PyZebArdYolo vs ZebTrack ──────────────────────────────
fa <- file.path(PARES, "pares_PyZebArdYolo.csv"); fb <- file.path(PARES, "pares_ZebTrack.csv")
if (file.exists(fa) && file.exists(fb)) {
  d <- rbind(read.csv(fa), read.csv(fb)); d <- d[!is.na(d$radial), ]
  d$metodo <- relevel(factor(d$metodo), ref = "PyZebArdYolo")
  cat("\n", strrep("=", 70), "\n", sep = "", file = out)
  cat("COMPARISON: ZebTrack vs PyZebArdYolo — radial ~ method + (1|video_id)\n", file = out)
  cat(strrep("=", 70), "\n", sep = "", file = out)
  mod <- lmer(radial ~ metodo + (1|video_id), data = d, REML = FALSE)
  b <- fixef(mod)["metodoZebTrack"]; se <- sqrt(diag(vcov(mod)))["metodoZebTrack"]
  cat(sprintf("  beta(ZebTrack) = %+.2f px  95%%CI[%.2f, %.2f]  t = %.1f\n",
              b, b - 1.96*se, b + 1.96*se, b/se), file = out)
}
close(out)
cat("Done. See results/mixed_model.txt\n")
