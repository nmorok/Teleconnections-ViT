# ==============================================================================
# QC: sum-per-year-per-bootstrap sanity check for raw Gaussian checkpoints
#
# Reads checkpoint_{spawner,recruit}_gauss_b###.rds directly from CHECKPOINT_DIR
# (35 survey years x 50x50 grid per file, one file per bootstrap) and plots
# spatial-sum-by-year for every bootstrap, so gaps/outliers/zero-runs are
# visible before running checkpoints_to_grid.R / training.
# ==============================================================================

CHECKPOINT_DIR <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output/pub/Gaussian/300"
OUT_DIR        <- CHECKPOINT_DIR
N_BOOTSTRAPS   <- 100

all_years    <- 1988:2023
survey_years <- all_years[all_years != 2020]   # 35 years, matches checkpoint contents

summarize_stage <- function(stage) {
  sums    <- matrix(NA_real_, nrow = N_BOOTSTRAPS, ncol = length(survey_years))
  maxes   <- matrix(NA_real_, nrow = N_BOOTSTRAPS, ncol = length(survey_years))
  n_neg   <- matrix(0L,       nrow = N_BOOTSTRAPS, ncol = length(survey_years))
  n_na    <- matrix(0L,       nrow = N_BOOTSTRAPS, ncol = length(survey_years))
  missing_files <- c()

  for (b in seq_len(N_BOOTSTRAPS)) {
    f <- file.path(CHECKPOINT_DIR, sprintf("checkpoint_%s_gauss_b%03d.rds", stage, b))
    if (!file.exists(f)) {
      missing_files <- c(missing_files, f)
      next
    }
    arr <- readRDS(f)  # expected [n_survey_yrs, 50, 50]
    if (length(dim(arr)) != 3 || dim(arr)[1] != length(survey_years)) {
      cat(sprintf("  [%s b%03d] UNEXPECTED SHAPE: %s\n", stage, b, paste(dim(arr), collapse="x")))
    }
    for (y in seq_len(dim(arr)[1])) {
      slice <- arr[y, , ]
      n_na[b, y]  <- sum(is.na(slice))
      n_neg[b, y] <- sum(slice < 0, na.rm = TRUE)
      sums[b, y]  <- sum(slice, na.rm = TRUE)
      maxes[b, y] <- max(slice, na.rm = TRUE)
    }
  }

  list(stage = stage, sums = sums, maxes = maxes, n_neg = n_neg, n_na = n_na,
       missing_files = missing_files)
}

report_issues <- function(res) {
  cat(sprintf("\n=== %s ===\n", res$stage))
  if (length(res$missing_files) > 0) {
    cat(sprintf("  MISSING checkpoint files (%d):\n", length(res$missing_files)))
    for (f in res$missing_files) cat("   ", f, "\n")
  } else {
    cat("  All 100 checkpoint files present.\n")
  }

  total_na  <- sum(res$n_na)
  total_neg <- sum(res$n_neg)
  cat(sprintf("  Total NA cells across all boot x year: %d\n", total_na))
  cat(sprintf("  Total negative cells across all boot x year: %d\n", total_neg))

  # Flag zero or near-zero sums (potential gaps) per bootstrap x year
  zero_cells <- which(res$sums == 0, arr.ind = TRUE)
  if (nrow(zero_cells) > 0) {
    cat(sprintf("  Bootstrap x Year combos with EXACT ZERO total sum (%d):\n", nrow(zero_cells)))
    for (i in seq_len(min(nrow(zero_cells), 30))) {
      b <- zero_cells[i, 1]; y <- zero_cells[i, 2]
      cat(sprintf("    b%03d  year=%d\n", b, survey_years[y]))
    }
    if (nrow(zero_cells) > 30) cat(sprintf("    ... and %d more\n", nrow(zero_cells) - 30))
  } else {
    cat("  No exact-zero-sum bootstrap x year combos.\n")
  }

  # Flag extreme outliers: sums > 5x the median for that year, across bootstraps
  med_by_year <- apply(res$sums, 2, median, na.rm = TRUE)
  for (y in seq_along(survey_years)) {
    outlier_b <- which(res$sums[, y] > 5 * med_by_year[y] & med_by_year[y] > 0)
    if (length(outlier_b) > 0) {
      cat(sprintf("  Year %d: outlier bootstraps (>5x median) -> %s\n",
                  survey_years[y], paste(sprintf("b%03d", outlier_b), collapse=", ")))
    }
  }
}

plot_stage <- function(res) {
  png(file.path(OUT_DIR, sprintf("qc_%s_sum_by_year.png", res$stage)),
      width = 1400, height = 900, res = 150)

  ylim <- range(res$sums, na.rm = TRUE)
  matplot(survey_years, t(res$sums), type = "l", lty = 1, col = rgb(0,0,1,0.15),
          xlab = "Year", ylab = "Spatial sum",
          main = sprintf("%s: spatial sum by year (100 bootstraps, gray=mean, red dashed=median)", res$stage),
          ylim = ylim)
  lines(survey_years, colMeans(res$sums, na.rm = TRUE), col = "black", lwd = 2)
  lines(survey_years, apply(res$sums, 2, median, na.rm = TRUE), col = "red", lwd = 2, lty = 2)
  abline(v = 2020, col = "darkgray", lty = 3)

  dev.off()
  cat(sprintf("  Wrote %s\n", file.path(OUT_DIR, sprintf("qc_%s_sum_by_year.png", res$stage))))
}

spawner_res <- summarize_stage("spawner")
recruit_res <- summarize_stage("recruit")

report_issues(spawner_res)
report_issues(recruit_res)

plot_stage(spawner_res)
plot_stage(recruit_res)

# Save the raw sums matrices too, in case you want to inspect them in Python/R later
saveRDS(spawner_res$sums, file.path(OUT_DIR, "qc_spawner_sums.rds"))
saveRDS(recruit_res$sums, file.path(OUT_DIR, "qc_recruit_sums.rds"))
write.csv(spawner_res$sums, file.path(OUT_DIR, "qc_spawner_sums.csv"), row.names = FALSE)
write.csv(recruit_res$sums, file.path(OUT_DIR, "qc_recruit_sums.csv"), row.names = FALSE)

cat("\nDone.\n")
