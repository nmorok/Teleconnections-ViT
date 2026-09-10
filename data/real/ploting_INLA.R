# ==============================================================================
# COMPARE TWEEDIE vs GAUSSIAN INLA GRIDDED OUTPUTS
#
# Assumes both pipelines have been run and outputs are on disk.
# Edit the two path constants below to point at your output directories.
#
# Arrays expected shape: [n_bootstraps, n_years, PAD_NY, PAD_NX]
# years.rds: integer vector length n_years (includes 2020)
# year_mask.rds: 0/1 vector (0 = 2020, no survey)
# spatial_mask.rds: PAD_NY x PAD_NX matrix (1 = valid cell)
# ==============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)    # install.packages("patchwork") if needed

# ── EDIT THESE ────────────────────────────────────────────────────────────────
TWEEDIE_DIR  <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output/Tweedie"     # directory with gridded_spawners.rds etc from Tweedie run
GAUSSIAN_DIR <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output/Gaussian/250"    # directory with equivalent files from Gaussian run
PLOT_DIR     <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output/plots"   # where to write PNGs
STAGE        <- "recruit"            # "spawner" or "recruit" — change and rerun for each
# ─────────────────────────────────────────────────────────────────────────────

dir.create(PLOT_DIR, showWarnings = FALSE)

# ── LOAD ARRAYS ───────────────────────────────────────────────────────────────
cat("Loading arrays...\n")

# Tweedie
tw_grids  <- readRDS(file.path(TWEEDIE_DIR,  paste0("gridded_", STAGE, "s.rds")))
tw_mask   <- readRDS(file.path(TWEEDIE_DIR,  "spatial_mask.rds"))
tw_years  <- readRDS(file.path(TWEEDIE_DIR,  "years.rds"))
tw_ymask  <- readRDS(file.path(TWEEDIE_DIR,  "year_mask.rds"))

# Gaussian — may have been saved under the same filenames in a different dir
ga_grids  <- readRDS(file.path(GAUSSIAN_DIR, paste0("gridded_", STAGE, "s.rds")))
ga_mask   <- readRDS(file.path(GAUSSIAN_DIR, "spatial_mask.rds"))
ga_years  <- readRDS(file.path(GAUSSIAN_DIR, "years.rds"))
ga_ymask  <- readRDS(file.path(GAUSSIAN_DIR, "year_mask.rds"))
#ga_mask <- tw_mask
#ga_years <- tw_years
#ga_ymask <- tw_ymask

stopifnot(all(tw_years == ga_years))   # years must align
years    <- tw_years
ymask    <- tw_ymask   # same for both
n_boot   <- dim(tw_grids)[1]
n_years  <- dim(tw_grids)[2]
PAD_NY   <- dim(tw_grids)[3]
PAD_NX   <- dim(tw_grids)[4]

cat(sprintf("  Array shape: [%d bootstraps x %d years x %d x %d]\n",
            n_boot, n_years, PAD_NY, PAD_NX))
cat(sprintf("  Years: %d-%d  (2020 masked)\n", min(years), max(years)))
cat(sprintf("  Valid cells: %d / %d\n", sum(tw_mask), PAD_NY * PAD_NX))

# Survey years only
survey_idx <- which(ymask == 1)
survey_yrs <- years[survey_idx]

# Spatial mask as vector index
valid_cells <- which(tw_mask == 1)   # length = n_valid

# ── HELPER: flatten grid -> valid cells ───────────────────────────────────────
# Returns [n_boot x n_years x n_valid] array
extract_valid <- function(grids, mask) {
  nb <- dim(grids)[1]; ny <- dim(grids)[2]
  nv <- sum(mask)
  out <- array(0, c(nb, ny, nv))
  for (b in seq_len(nb))
    for (t in seq_len(ny))
      out[b, t, ] <- grids[b, t, , ][mask == 1]
  out
}

cat("Extracting valid cells...\n")
tw_valid <- extract_valid(tw_grids, tw_mask)
ga_valid <- extract_valid(ga_grids, ga_mask)

tw_ymask <- tw_ymask[seq_along(tw_years)]
ga_ymask <- ga_ymask[seq_along(ga_years)]

# ── 1. TOTAL ABUNDANCE INDEX (sum over valid cells) per bootstrap x year ──────
cat("Computing total abundance index...\n")

tw_abund <- apply(tw_valid, c(1, 2), sum)   # [n_boot x n_years]
ga_abund <- apply(ga_valid, c(1, 2), sum)

# Summarise across bootstraps
abund_summary <- function(mat, yr_vec, ym_vec, method_label) {
  stopifnot(ncol(mat) == length(yr_vec), ncol(mat) == length(ym_vec))
  data.frame(
    year    = yr_vec,
    ymask   = ym_vec,
    mean    = apply(mat, 2, mean),
    med     = apply(mat, 2, median),
    log_med = apply(log(mat), 2, median),
    lo      = apply(mat, 2, quantile, 0.025),
    hi      = apply(mat, 2, quantile, 0.975),
    log_lo  = apply(log(mat), 2, quantile, 0.025),
    log_hi  = apply(log(mat), 2, quantile, 0.975),
    method  = method_label
  )
}

abund_df <- bind_rows(
  abund_summary(tw_abund, tw_years, tw_ymask, "Tweedie"),
  abund_summary(ga_abund, ga_years, ga_ymask, "Gaussian")
) %>% filter(ymask == 1)

p_abund <- ggplot(abund_df, aes(x = year, y = mean / 1e6, colour = method, fill = method)) +
  geom_ribbon(aes(ymin = lo / 1e6, ymax = hi / 1e6), alpha = 0.2, colour = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  scale_colour_manual(values = c(Tweedie = "#1b7837", Gaussian = "#762a83")) +
  scale_fill_manual(  values = c(Tweedie = "#1b7837", Gaussian = "#762a83")) +
  labs(
    title  = paste(tools::toTitleCase(STAGE), "— Total Abundance Index"),
    subtitle = "Mean ± 95% bootstrap CI  |  sum over valid grid cells",
    x = "Year", y = "Total density (millions)", colour = NULL, fill = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(PLOT_DIR, paste0(STAGE, "_01_abundance_index.png")),
       p_abund, width = 10, height = 4.5, dpi = 150)
cat("  Saved abundance index plot\n")

p_abund_log <- ggplot(abund_df, aes(x = year, y = log_med, colour = method, fill = method)) +
  geom_ribbon(aes(ymin = log_lo, ymax = log_hi), alpha = 0.2, colour = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  scale_colour_manual(values = c(Tweedie = "#1b7837", Gaussian = "#762a83")) +
  scale_fill_manual(  values = c(Tweedie = "#1b7837", Gaussian = "#762a83")) +
  labs(
    title    = paste(tools::toTitleCase(STAGE), "— Total Abundance Index (log scale)"),
    subtitle = "Median ± 95% bootstrap CI  |  computed on log scale",
    x = "Year", y = "log(Total density)", colour = NULL, fill = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "top")


# --- Build raw bootstrap long data ---
boot_to_long <- function(mat, method_label, yr_vec) {
  # yr_vec is the full years vector (length == ncol(mat))
  stopifnot(ncol(mat) == length(yr_vec))
  colnames(mat) <- as.character(yr_vec)
  as.data.frame(mat) %>%
    mutate(boot = row_number()) %>%
    pivot_longer(-boot, names_to = "year", values_to = "abund") %>%
    mutate(year = as.integer(year), method = method_label)
}

boot_df <- bind_rows(
  boot_to_long(tw_abund, "Tweedie",  years),
  boot_to_long(ga_abund, "Gaussian", years)
) %>%
  filter(year %in% years[ymask == 1])

# --- Median summary (already in abund_df, just make clear) ---
# abund_df$med is the per-year median across bootstraps

# --- Plot ---
p_abund_boot <- ggplot() +
  # Individual bootstrap lines (gray, transparent)
  geom_line(
    data = boot_df,
    aes(x = year, y = abund / 1e6, group = interaction(boot, method), colour = method),
    alpha = 0.3, linewidth = 0.5
  ) +
  # Median line (bold)
  geom_line(
    data = abund_df,
    aes(x = year, y = med / 1e6, colour = method),
    linewidth = 1.2
  ) +
  geom_point(
    data = abund_df,
    aes(x = year, y = med / 1e6, colour = method),
    size = 1.8
  ) +
  scale_colour_manual(values = c(Tweedie = "#1b7837", Gaussian = "#762a83")) +
  labs(
    title    = paste(tools::toTitleCase(STAGE), "— Total Abundance Index"),
    subtitle = "Individual bootstrap runs (faint) + median (bold)",
    x = "Year", y = "Total density (millions)", colour = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "top")

p_abund_boot_log <- ggplot() +
  geom_line(
    data = boot_df,
    aes(x = year, y = log(abund), group = interaction(boot, method), colour = method),
    alpha = 0.3, linewidth = 0.5
  ) +
  geom_line(
    data = abund_df,
    aes(x = year, y = log_med, colour = method),   # ← median of logs, not log of median
    linewidth = 1.2
  ) +
  geom_point(
    data = abund_df,
    aes(x = year, y = log_med, colour = method),
    size = 1.8
  ) +
  scale_colour_manual(values = c(Tweedie = "#1b7837", Gaussian = "#762a83")) +
  labs(
    title    = paste(tools::toTitleCase(STAGE), "— Total log(Abundance) Index"),
    subtitle = "Individual bootstrap runs (faint) + median of log(abundance) (bold)",
    x = "Year", y = "log(Total density)", colour = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "top")



# ── 2. BOOTSTRAP VARIANCE per year ────────────────────────────────────────────
cat("Computing bootstrap variance by year...\n")

# CV of total abundance across bootstraps
cv_df <- data.frame(
  year    = years[survey_idx],
  cv_tw   = apply(tw_abund[, survey_idx], 2, sd) / apply(tw_abund[, survey_idx], 2, mean),
  cv_ga   = apply(ga_abund[, survey_idx], 2, sd) / apply(ga_abund[, survey_idx], 2, mean)
) %>%
  pivot_longer(c(cv_tw, cv_ga), names_to = "method", values_to = "cv") %>%
  mutate(method = recode(method, cv_tw = "Tweedie", cv_ga = "Gaussian"))

cv_df_wide <- cv_df %>%
  pivot_wider(names_from = method, values_from = cv)

p_cv <- ggplot(cv_df, aes(x = year, y = cv, colour = method)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  scale_colour_manual(values = c(Tweedie = "#1b7837", Gaussian = "#762a83")) +
  labs(
    title    = paste(tools::toTitleCase(STAGE), "— Bootstrap CV of Total Abundance"),
    subtitle = "sd / mean across 100 bootstraps",
    x = "Year", y = "CV", colour = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(PLOT_DIR, paste0(STAGE, "_02_bootstrap_cv.png")),
       p_cv, width = 10, height = 4, dpi = 150)
cat("  Saved CV plot\n")

# ── 3. NEGATIVE PREDICTION FRACTION per year (Gaussian artifact) ───────────────
cat("Checking negative predictions...\n")

neg_frac <- data.frame(
  year   = years[survey_idx],
  tw_neg = apply(tw_valid[, survey_idx, ], 2,
                 function(mat) mean(mat < 0)),
  ga_neg = apply(ga_valid[, survey_idx, ], 2,
                 function(mat) mean(mat < 0))
)

if (any(neg_frac$ga_neg > 0) || any(neg_frac$tw_neg > 0)) {
  neg_long <- neg_frac %>%
    pivot_longer(c(tw_neg, ga_neg), names_to = "method", values_to = "neg_frac") %>%
    mutate(method = recode(method, tw_neg = "Tweedie", ga_neg = "Gaussian"))
  
  p_neg <- ggplot(neg_long, aes(x = year, y = neg_frac * 100, colour = method)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.5) +
    scale_colour_manual(values = c(Tweedie = "#1b7837", Gaussian = "#762a83")) +
    labs(
      title    = paste(tools::toTitleCase(STAGE), "— Fraction of Negative Predictions"),
      subtitle = "% of bootstrap x cell predictions < 0  (Tweedie should be ~0)",
      x = "Year", y = "% negative", colour = NULL
    ) +
    theme_bw(base_size = 12) +
    theme(legend.position = "top")
  
  ggsave(file.path(PLOT_DIR, paste0(STAGE, "_03_negative_preds.png")),
         p_neg, width = 10, height = 4, dpi = 150)
  cat("  Saved negative predictions plot\n")
} else {
  cat("  No negative predictions in either model\n")
}

cat("\nNegative prediction summary:\n")
print(neg_frac)

# ── 4. MEAN SPATIAL MAPS — selected years ─────────────────────────────────────
cat("Generating spatial comparison maps...\n")

# Pick 6 representative years
highlight_years <- c(1990, 2000, 2005, 2010, 2018, 2022)
highlight_years <- highlight_years[highlight_years %in% survey_yrs]

mask_df <- expand.grid(row = seq_len(PAD_NY), col = seq_len(PAD_NX)) %>%
  mutate(valid = as.vector(tw_mask) == 1)

for (yr in highlight_years) {
  yr_idx <- which(years == yr)
  
  # Bootstrap mean map for this year
  tw_map_vec <- apply(tw_grids[, yr_idx, , ], c(2, 3), mean)
  ga_map_vec <- apply(ga_grids[, yr_idx, , ], c(2, 3), mean)
  
  make_map_df <- function(mat, method_label) {
    expand.grid(row = seq_len(PAD_NY), col = seq_len(PAD_NX)) %>%
      mutate(
        density = as.vector(mat),
        method  = method_label
      ) %>%
      left_join(mask_df, by = c("row", "col")) %>%
      mutate(density = ifelse(valid, density, NA_real_))
  }
  
  map_df <- bind_rows(
    make_map_df(tw_map_vec, "Tweedie"),
    make_map_df(ga_map_vec, "Gaussian")
  )
  
  # Shared colour scale (99th pct of both, ignore NAs)
  clim <- quantile(map_df$density, 0.99, na.rm = TRUE)
  
  p_map <- ggplot(map_df, aes(x = col, y = rev(row), fill = pmin(density, clim))) +
    geom_raster() +
    facet_wrap(~method) +
    scale_fill_viridis_c(
      option   = "plasma",
      na.value = "grey90",
      name     = "Density\n(clipped 99%)",
      limits   = c(0, clim)
    ) +
    coord_equal() +
    labs(
      title    = paste(tools::toTitleCase(STAGE), "— Mean Predicted Density,", yr),
      subtitle = "Bootstrap mean across 100 replicates"
    ) +
    theme_void(base_size = 11) +
    theme(
      strip.text    = element_text(face = "bold"),
      plot.title    = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5, size = 9),
      legend.position = "right"
    )
  
  fname <- file.path(PLOT_DIR, sprintf("%s_04_map_%d.png", STAGE, yr))
  ggsave(fname, p_map, width = 10, height = 5, dpi = 150)
}
cat(sprintf("  Saved maps for years: %s\n", paste(highlight_years, collapse = ", ")))

# ── 5. DIFFERENCE MAP — Tweedie minus Gaussian, mean across years ─────────────
cat("Generating difference maps...\n")

# Mean across bootstraps and survey years
tw_mean_all <- apply(tw_grids[, survey_idx, , ], c(2, 3), mean)  # [PAD_NY x PAD_NX] avg over b
ga_mean_all <- apply(ga_grids[, survey_idx, , ], c(2, 3), mean)


# Mean across bootstraps (dim 1) and survey years (dim 2), keeping spatial dims 3 & 4
tw_grand <- apply(tw_grids[, survey_idx, , ], c(3, 4), mean)  # [PAD_NY x PAD_NX]
ga_grand <- apply(ga_grids[, survey_idx, , ], c(3, 4), mean)
diff_grand <- tw_grand - ga_grand

diff_df <- expand.grid(row = seq_len(PAD_NY), col = seq_len(PAD_NX)) %>%
  mutate(
    diff  = as.vector(diff_grand),
    valid = as.vector(tw_mask) == 1,
    diff  = ifelse(valid, diff, NA_real_)
  )

clim_diff <- max(abs(quantile(diff_df$diff, c(0.01, 0.99), na.rm = TRUE)))

p_diff <- ggplot(diff_df, aes(x = col, y = rev(row), fill = diff)) +
  geom_raster() +
  scale_fill_gradient2(
    low      = "#762a83",
    mid      = "white",
    high     = "#1b7837",
    midpoint = 0,
    limits   = c(-clim_diff, clim_diff),
    na.value = "grey90",
    name     = "Tweedie −\nGaussian"
  ) +
  coord_equal() +
  labs(
    title    = paste(tools::toTitleCase(STAGE), "— Grand Mean Difference (Tweedie − Gaussian)"),
    subtitle = "Averaged over all survey years and bootstraps"
  ) +
  theme_void(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 9))

ggsave(file.path(PLOT_DIR, paste0(STAGE, "_05_diff_map.png")),
       p_diff, width = 6, height = 6, dpi = 150)
cat("  Saved difference map\n")

# ── 6. CELL-LEVEL SCATTER — Tweedie vs Gaussian (bootstrap mean, all yr x cell) 
cat("Generating cell-level scatter...\n")

# Subsample to keep plot manageable: 5 years x all valid cells
scatter_yrs <- round(seq(1, length(survey_idx), length.out = 5))
scatter_idx <- survey_idx[scatter_yrs]

scatter_df <- do.call(rbind, lapply(scatter_idx, function(ti) {
  yr <- years[ti]
  data.frame(
    year    = yr,
    tw_mean = apply(tw_valid[, ti, ], 2, mean),
    ga_mean = apply(ga_valid[, ti, ], 2, mean)
  )
}))

p_scatter <- ggplot(scatter_df, aes(x = ga_mean, y = tw_mean)) +
  geom_hex(bins = 60) +
  geom_abline(slope = 1, intercept = 0, colour = "red", linetype = "dashed") +
  scale_fill_viridis_c(option = "magma", trans = "log10", name = "# cells") +
  facet_wrap(~year, scales = "free") +
  labs(
    title    = paste(tools::toTitleCase(STAGE), "— Cell-level: Tweedie vs Gaussian Bootstrap Means"),
    subtitle = "Red dashed = 1:1 line; one point per valid grid cell",
    x = "Gaussian mean", y = "Tweedie mean"
  ) +
  theme_bw(base_size = 11) +
  theme(strip.text = element_text(face = "bold"))

ggsave(file.path(PLOT_DIR, paste0(STAGE, "_06_cell_scatter.png")),
       p_scatter, width = 12, height = 8, dpi = 150)
cat("  Saved cell scatter plot\n")

# ── 7. PRINTED SUMMARY TABLE ──────────────────────────────────────────────────
cat("\n====== SUMMARY STATISTICS ======\n\n")

# Grand mean total abundance
cat(sprintf("Grand mean total abundance (survey years only):\n"))
cat(sprintf("  Tweedie  : %.1f  (median: %.1f)\n",
            mean(tw_abund[, survey_idx]),
            median(tw_abund[, survey_idx])))
cat(sprintf("  Gaussian : %.1f  (median: %.1f)\n",
            mean(ga_abund[, survey_idx]),
            median(ga_abund[, survey_idx])))
cat(sprintf("  Ratio (Tw/Ga): %.3f\n\n",
            mean(tw_abund[, survey_idx]) / mean(ga_abund[, survey_idx])))

# Fraction of negative predictions overall
tw_neg_all <- mean(tw_valid[, survey_idx, ] < 0)
ga_neg_all <- mean(ga_valid[, survey_idx, ] < 0)
cat(sprintf("Fraction of negative cell-bootstrap predictions:\n"))
cat(sprintf("  Tweedie  : %.4f%%\n", tw_neg_all * 100))
cat(sprintf("  Gaussian : %.4f%%\n", ga_neg_all * 100))

# Bootstrap CV averaged across years
tw_cv_mean <- mean(apply(tw_abund[, survey_idx], 2, sd) /
                     apply(tw_abund[, survey_idx], 2, mean))
ga_cv_mean <- mean(apply(ga_abund[, survey_idx], 2, sd) /
                     apply(ga_abund[, survey_idx], 2, mean))
cat(sprintf("\nMean bootstrap CV of total abundance:\n"))
cat(sprintf("  Tweedie  : %.4f\n", tw_cv_mean))
cat(sprintf("  Gaussian : %.4f\n", ga_cv_mean))

cat(sprintf("\nAll plots saved to: %s/\n", PLOT_DIR))
cat("Files:\n")
for (f in list.files(PLOT_DIR, pattern = paste0("^", STAGE, "_"), full.names = FALSE))
  cat(sprintf("  %s\n", f))
