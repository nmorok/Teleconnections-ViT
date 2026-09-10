# ==============================================================================
# INLA Comparison Plot: Observed Stations vs. Gridded Field
#
# PREREQUISITES: pipeline.R must have been sourced so these objects exist:
#   spawner_sf, recruit_sf, station_locations_sf, survey_domain, sf_maps
#
# USAGE: Source this file after pipeline.R, or run interactively.
# ==============================================================================

library(INLA)
library(fmesher)
library(sf)
library(dplyr)
library(ggplot2)
library(patchwork)

# ==============================================================================
# ── CONFIGURATION (edit these) ─────────────────────────────────────────────────
# ==============================================================================

PLOT_YEAR      <- 2005         # year to visualize (must be in spawner/recruit data)
VARIABLE       <- "spawner"    # "spawner" or "recruit"
BOOTSTRAP_IDX  <- "mean"       # integer (e.g. 1) or "mean" to average all bootstraps

OUTPUT_DIR     <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output"
SAVE_PLOT      <- TRUE         # write PNG to OUTPUT_DIR?
PLOT_FILENAME  <- NULL         # NULL → auto-named; or e.g. "comparison_2005.png"

# Color scale limits — NULL = auto-fit to data range
OBS_COLOR_LIMITS  <- NULL      # e.g. c(0, 10) for log1p(avg_dens)
GRID_COLOR_LIMITS <- NULL      # e.g. c(0, 10) for log1p(gridded)

# Point size for station dots
POINT_SIZE <- 3.5

# Figure dimensions (inches)
FIG_WIDTH  <- 14
FIG_HEIGHT <- 6

# ==============================================================================
# ── HELPERS ────────────────────────────────────────────────────────────────────
# ==============================================================================

# Rebuild the prediction grid (mirrors process_data.R)
CELLSIZE <- 25
PAD_NX   <- 50
PAD_NY   <- 50

build_grid_local <- function(survey_domain, station_locations_sf, cellsize = CELLSIZE) {
  full_grid <- st_make_grid(survey_domain, cellsize = cellsize,
                            crs = st_crs(survey_domain))
  hits <- st_intersects(full_grid, survey_domain, sparse = FALSE)[, 1]
  stn_hits <- st_intersects(full_grid, station_locations_sf, sparse = FALSE)
  hits <- hits | (rowSums(stn_hits) > 0)
  valid_grid <- full_grid[hits]
  valid_idx  <- which(hits)

  bbox   <- st_bbox(survey_domain)
  nx_full <- ceiling((bbox["xmax"] - bbox["xmin"]) / cellsize)
  ny_full <- ceiling((bbox["ymax"] - bbox["ymin"]) / cellsize)

  grid_col <- ((valid_idx - 1) %% nx_full) + 1
  grid_row <- ((valid_idx - 1) %/% nx_full) + 1
  grid_row <- ny_full + 1 - grid_row   # flip: 1 = north

  centroids <- st_coordinates(st_centroid(valid_grid))

  list(valid_grid = valid_grid, centroids = centroids,
       grid_col = grid_col, grid_row = grid_row,
       nx_full = nx_full, ny_full = ny_full,
       n_valid = length(valid_grid), cellsize = cellsize,
       bbox = bbox, valid_idx = valid_idx)
}

# Extract valid-cell values from a padded [50×50] matrix
extract_valid <- function(mat, grid_info) {
  vapply(seq_len(grid_info$n_valid),
         function(i) mat[grid_info$grid_row[i], grid_info$grid_col[i]],
         numeric(1))
}

# ==============================================================================
# ── LOAD DATA ──────────────────────────────────────────────────────────────────
# ==============================================================================

cat(sprintf("[plot] Variable: %s  |  Year: %d  |  Bootstrap: %s\n",
            VARIABLE, PLOT_YEAR, as.character(BOOTSTRAP_IDX)))

# Pick the right raw SF data and gridded RDS
if (VARIABLE == "spawner") {
  raw_sf    <- spawner_sf
  grid_file <- file.path(OUTPUT_DIR, "gridded_spawners.rds")
} else if (VARIABLE == "recruit") {
  raw_sf    <- recruit_sf
  grid_file <- file.path(OUTPUT_DIR, "gridded_recruits.rds")
} else {
  stop("VARIABLE must be 'spawner' or 'recruit'")
}

years_vec <- readRDS(file.path(OUTPUT_DIR, "years.rds"))   # 1988:2023 with 2020 gap
grids     <- readRDS(grid_file)                             # [n_boot, n_years, 50, 50]
n_boot    <- dim(grids)[1]

# Year index into the padded array (2020 slot exists but is zeros)
yr_idx_padded <- which(years_vec == PLOT_YEAR)
if (length(yr_idx_padded) == 0)
  stop(sprintf("Year %d not found in years.rds (%d–%d)", PLOT_YEAR,
               min(years_vec), max(years_vec)))

# Extract gridded field
if (identical(BOOTSTRAP_IDX, "mean")) {
  grid_mat <- apply(grids[, yr_idx_padded, , , drop = FALSE], c(3, 4), mean)
  boot_label <- "Mean (100 bootstraps)"
} else {
  if (BOOTSTRAP_IDX < 1 || BOOTSTRAP_IDX > n_boot)
    stop(sprintf("BOOTSTRAP_IDX %d out of range [1, %d]", BOOTSTRAP_IDX, n_boot))
  grid_mat  <- grids[BOOTSTRAP_IDX, yr_idx_padded, , ]
  boot_label <- sprintf("Bootstrap %d", BOOTSTRAP_IDX)
}

# ==============================================================================
# ── BUILD GRID GEOMETRY ────────────────────────────────────────────────────────
# ==============================================================================

cat("[plot] Building grid geometry...\n")

station_locations_sf_named <- station_locations_sf
if (!"station" %in% names(station_locations_sf_named))
  station_locations_sf_named <- station_locations_sf_named %>%
    rename(station = GIS_STATION)

grid_info <- build_grid_local(survey_domain, station_locations_sf_named)
valid_vals <- extract_valid(grid_mat, grid_info)

grid_sf <- st_sf(
  geometry = grid_info$valid_grid,
  density  = valid_vals
)

# ==============================================================================
# ── OBSERVED STATION DATA ──────────────────────────────────────────────────────
# ==============================================================================

obs_yr <- raw_sf %>% filter(year == PLOT_YEAR)

if (nrow(obs_yr) == 0)
  stop(sprintf("No %s observations found for year %d", VARIABLE, PLOT_YEAR))

cat(sprintf("[plot] Observed stations: %d  |  Gridded cells: %d\n",
            nrow(obs_yr), nrow(grid_sf)))

# ==============================================================================
# ── COLOR SCALE HELPERS ────────────────────────────────────────────────────────
# ==============================================================================

obs_vals  <- log1p(obs_yr$avg_dens)
grid_vals <- log1p(pmax(grid_sf$density, 0))

obs_limits  <- if (!is.null(OBS_COLOR_LIMITS))  OBS_COLOR_LIMITS  else range(obs_vals,  na.rm = TRUE)
grid_limits <- if (!is.null(GRID_COLOR_LIMITS)) GRID_COLOR_LIMITS else range(grid_vals, na.rm = TRUE)

# ==============================================================================
# ── PLOT 1: OBSERVED STATIONS ──────────────────────────────────────────────────
# ==============================================================================

p_obs <- ggplot() +
  geom_sf(data = sf_maps,     fill = "grey75", color = "grey55", linewidth = 0.3) +
  geom_sf(data = survey_domain, fill = NA,      color = "grey30", linewidth = 0.6) +
  geom_sf(data = obs_yr,
          aes(color = log1p(avg_dens)),
          size = POINT_SIZE, shape = 16) +
  scale_color_viridis_c(
    name   = "log1p(density)",
    limits = obs_limits,
    option = "plasma"
  ) +
  coord_sf(expand = FALSE) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(size = 12, face = "bold"),
    legend.position = "right"
  ) +
  labs(
    title    = sprintf("Observed — %s %d", tools::toTitleCase(VARIABLE), PLOT_YEAR),
    subtitle = sprintf("n = %d stations", nrow(obs_yr))
  )

# ==============================================================================
# ── PLOT 2: INLA GRIDDED FIELD ─────────────────────────────────────────────────
# ==============================================================================

p_grid <- ggplot() +
  geom_sf(data = grid_sf,
          aes(fill = log1p(pmax(density, 0))),
          color = NA) +
  geom_sf(data = sf_maps,     fill = "grey75", color = "grey55", linewidth = 0.3) +
  geom_sf(data = survey_domain, fill = NA,     color = "grey30", linewidth = 0.6) +
  scale_fill_viridis_c(
    name   = "log1p(density)",
    limits = grid_limits,
    option = "plasma"
  ) +
  coord_sf(expand = FALSE) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(size = 12, face = "bold"),
    legend.position = "right"
  ) +
  labs(
    title    = sprintf("INLA Grid — %s %d", tools::toTitleCase(VARIABLE), PLOT_YEAR),
    subtitle = boot_label
  )

# ==============================================================================
# ── COMBINE AND DISPLAY ────────────────────────────────────────────────────────
# ==============================================================================

combined <- p_obs + p_grid +
  plot_annotation(
    title   = sprintf("%s density: Observed vs. INLA grid (%d)",
                      tools::toTitleCase(VARIABLE), PLOT_YEAR),
    caption = "Color: log1p(density)"
  )

print(combined)

if (SAVE_PLOT) {
  fname <- if (!is.null(PLOT_FILENAME)) PLOT_FILENAME else
    sprintf("inla_comparison_%s_%d_%s.png", VARIABLE, PLOT_YEAR,
            ifelse(identical(BOOTSTRAP_IDX, "mean"), "mean",
                   sprintf("boot%02d", BOOTSTRAP_IDX)))
  out_path <- file.path(OUTPUT_DIR, fname)
  ggsave(out_path, combined, width = FIG_WIDTH, height = FIG_HEIGHT, dpi = 150)
  cat(sprintf("[plot] Saved → %s\n", out_path))
}
