# ==============================================================================
# NUMERICAL COMPARISON — Tweedie vs Gaussian INLA gridded fields
#
# Produces a table of per-year and global statistics to assess whether
# the two preprocessing choices produce meaningfully different training data.
#
# Run AFTER step0_assemble_tweedie.R and with ga_grids / tw_grids in session.
# ==============================================================================

library(dplyr)
library(tidyr)


# Fix year alignment — ga_grids has 35 years in dim 2 but ga_years has 36
# so drop 2020 from both
keep     <- tw_years != 2020
tw_grids <- tw_grids[, keep, , ]
tw_years <- tw_years[keep]

keep     <- ga_years != 2020
ga_grids <- ga_grids[, keep, , ]
ga_years <- ga_years[keep]

stopifnot(dim(tw_grids)[2] == 35, dim(ga_grids)[2] == 35)
years   <- tw_years
n_years <- length(years)
n_boot  <- dim(tw_grids)[1]
valid_cells <- which(tw_mask == 1)

cat(sprintf("Comparing: [%d x %d x 50 x 50] vs [%d x %d x 50 x 50]\n",
            n_boot, n_years, dim(ga_grids)[1], dim(ga_grids)[2]))

# Clean mean_fields function — no dead code
mean_fields <- function(grids) {
  sapply(seq_len(n_years), function(t) {
    m <- apply(grids[, t, , ], c(2, 3), mean)
    as.vector(m)[valid_cells]
  }) |> t()   # [n_years x n_valid]
}

tw_fields <- mean_fields(tw_grids)
ga_fields <- mean_fields(ga_grids)
cat(sprintf("Fields shape: [%d x %d]\n", nrow(tw_fields), ncol(tw_fields)))



# ── ALIGN YEARS ───────────────────────────────────────────────────────────────
# Drop 2020 from Tweedie if Gaussian doesn't have it
if (dim(tw_grids)[2] != dim(ga_grids)[2]) {
  keep      <- tw_years != 2020
  tw_grids  <- tw_grids[, keep, , ]
  tw_years  <- tw_years[keep]
}
stopifnot(length(tw_years) == length(ga_years))
years      <- tw_years   # 35 survey years
n_years    <- length(years)
n_boot     <- dim(tw_grids)[1]
valid_cells <- which(tw_mask == 1)   # integer index into flattened 50x50

cat(sprintf("Comparing arrays: [%d bootstraps x %d years x 50 x 50]\n",
            n_boot, n_years))
cat(sprintf("Valid cells: %d\n\n", length(valid_cells)))


# ── 1. PER-YEAR SPEARMAN + PEARSON + RMSE + BIAS
year_stats <- do.call(rbind, lapply(seq_len(n_years), function(t) {
  tw <- tw_fields[t, ]
  ga <- ga_fields[t, ]
  ga_pos <- pmax(ga, 0)
  tw_pos <- pmax(tw, 0)
  data.frame(
    year         = years[t],
    spearman     = cor(tw, ga, method = "spearman"),
    pearson      = cor(tw, ga, method = "pearson"),
    rmse         = sqrt(mean((tw - ga)^2)),
    mae          = mean(abs(tw - ga)),
    bias         = mean(tw - ga),
    ratio_means  = mean(tw_pos) / (mean(ga_pos) + 1e-9),
    log_spearman = cor(log1p(tw_pos), log1p(ga_pos), method = "spearman"),
    log_pearson  = cor(log1p(tw_pos), log1p(ga_pos), method = "pearson"),
    log_rmse     = sqrt(mean((log1p(tw_pos) - log1p(ga_pos))^2)),
    hotspot_agree = mean((tw >= quantile(tw, 0.75)) == (ga >= quantile(ga, 0.75))),
    tw_neg_frac  = mean(tw < 0),
    ga_neg_frac  = mean(ga < 0),
    tw_total     = sum(tw_pos),
    ga_total     = sum(ga_pos)
  )
}))

cat("\n====== PER-YEAR STATISTICS ======\n\n")
print(year_stats, digits = 3, row.names = FALSE)

# ── 2. GLOBAL SUMMARY
cat("\n====== GLOBAL SUMMARY ======\n\n")
cat(sprintf("Spearman (raw)      — mean: %.4f  min: %.4f  max: %.4f\n",
            mean(year_stats$spearman), min(year_stats$spearman), max(year_stats$spearman)))
cat(sprintf("Pearson  (raw)      — mean: %.4f  min: %.4f  max: %.4f\n",
            mean(year_stats$pearson), min(year_stats$pearson), max(year_stats$pearson)))
cat(sprintf("Spearman (log1p)    — mean: %.4f  min: %.4f  max: %.4f\n",
            mean(year_stats$log_spearman), min(year_stats$log_spearman), max(year_stats$log_spearman)))
cat(sprintf("Pearson  (log1p)    — mean: %.4f  min: %.4f  max: %.4f\n",
            mean(year_stats$log_pearson), min(year_stats$log_pearson), max(year_stats$log_pearson)))
cat(sprintf("Log RMSE            — mean: %.4f  min: %.4f  max: %.4f\n",
            mean(year_stats$log_rmse), min(year_stats$log_rmse), max(year_stats$log_rmse)))
cat(sprintf("Hotspot agreement   — mean: %.4f  min: %.4f  max: %.4f\n",
            mean(year_stats$hotspot_agree), min(year_stats$hotspot_agree), max(year_stats$hotspot_agree)))
cat(sprintf("Ratio of means Tw/Ga— mean: %.4f  min: %.4f  max: %.4f\n",
            mean(year_stats$ratio_means), min(year_stats$ratio_means), max(year_stats$ratio_means)))
cat(sprintf("Gaussian neg frac   — mean: %.4f%%  max: %.4f%%\n",
            mean(year_stats$ga_neg_frac)*100, max(year_stats$ga_neg_frac)*100))
cat(sprintf("Tweedie  neg frac   — mean: %.4f%%\n",
            mean(year_stats$tw_neg_frac)*100))

# ── 3. BOOTSTRAP CV
tw_totals <- sapply(seq_len(n_years), function(t)
  apply(tw_grids[, t, , ], 1, function(x) sum(pmax(as.vector(x)[valid_cells], 0))))
ga_totals <- sapply(seq_len(n_years), function(t)
  apply(ga_grids[, t, , ], 1, function(x) sum(pmax(as.vector(x)[valid_cells], 0))))

cv_df <- data.frame(
  year    = years,
  tw_cv   = apply(tw_totals, 2, sd) / apply(tw_totals, 2, mean),
  ga_cv   = apply(ga_totals, 2, sd) / apply(ga_totals, 2, mean),
  tw_mean = apply(tw_totals, 2, mean),
  ga_mean = apply(ga_totals, 2, mean)
)
cv_df$cv_ratio <- cv_df$tw_cv / cv_df$ga_cv

cat("\n====== BOOTSTRAP CV ======\n\n")
print(cv_df, digits = 3, row.names = FALSE)
cat(sprintf("\nMean CV — Tweedie: %.4f  Gaussian: %.4f  (ratio: %.3f)\n",
            mean(cv_df$tw_cv), mean(cv_df$ga_cv),
            mean(cv_df$tw_cv) / mean(cv_df$ga_cv)))






boot_stats <- do.call(rbind, lapply(seq_len(n_years), function(t) {
  metrics <- do.call(rbind, lapply(seq_len(n_boot), function(b) {
    tw <- pmax(as.vector(tw_grids[b, t, , ])[valid_cells], 0)
    ga <- pmax(as.vector(ga_grids[b, t, , ])[valid_cells], 0)
    data.frame(
      spearman = cor(tw, ga, method = "spearman", use = "complete.obs"),
      mae      = mean(abs(tw - ga), na.rm = TRUE)
    )
  }))
  data.frame(
    year         = years[t],
    spearman     = mean(metrics$spearman, na.rm = TRUE),
    spearman_sd  = sd(metrics$spearman,   na.rm = TRUE),
    mae          = mean(metrics$mae,       na.rm = TRUE),
    mae_sd       = sd(metrics$mae,         na.rm = TRUE)
  )
}))

sum_stats <- do.call(rbind, lapply(seq_len(n_years), function(t) {
  sums <- do.call(rbind, lapply(seq_len(n_boot), function(b) {
    tw <- pmax(as.vector(tw_grids[b, t, , ])[valid_cells], 0)
    ga <- pmax(as.vector(ga_grids[b, t, , ])[valid_cells], 0)
    data.frame(
      tw_sum = sum(tw, na.rm = TRUE),
      ga_sum = sum(ga, na.rm = TRUE)
    )
  }))
  data.frame(
    year   = years[t],
    mae    = mean(abs(sums$tw_sum - sums$ga_sum), na.rm = TRUE),
    mae_sd = sd(abs(sums$tw_sum - sums$ga_sum),   na.rm = TRUE)
  )
}))
