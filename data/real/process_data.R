#' ==============================================================================
#' SPDE-Based Gridding of EBS Survey Data with Bootstrap Subsampling
#' ==============================================================================
#'
#' This script:
#'   1. Builds a regular 50x35 prediction grid over the EBS survey domain
#'   2. Constructs an SPDE mesh from station locations
#'   3. For each of 100 bootstraps, subsamples 300/349 stations
#'   4. Fits an INLA SPDE model per year on the subsampled stations
#'   5. Projects predictions onto the regular grid
#'   6. Saves gridded arrays as .npy for Python consumption
#'   7. Saves spatial mask (which grid cells are inside the survey domain)
#'
#' Output:
#'   gridded_spawners.npy  — array [n_bootstraps, n_years, 50, 50] (padded)
#'   gridded_recruits.npy  — array [n_bootstraps, n_years, 50, 50] (padded)
#'   spatial_mask.npy       — array [50, 50] binary mask
#'   grid_metadata.json     — grid parameters for Python
#'
#' Prerequisites:
#'   - Run pipeline.R first to get data_list, survey_domain, etc.
#'   - Or source this after pipeline.R
#'
#' Runtime estimate: ~3-8 hours for 100 bootstraps x 30 years x 2 variables
#'   Adjust N_BOOTSTRAPS for testing (e.g., set to 5 for a quick check)
#' ==============================================================================

library(INLA)
library(fmesher)
library(sf)
library(dplyr)
library(jsonlite)

# Optional: for saving .npy directly from R
# install.packages("reticulate")
# library(reticulate)
# np <- import("numpy")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CELLSIZE      <- 25
GRID_NX       <- 50    # columns (E-W, ~23 km cells)
GRID_NY       <- 35# rows (N-S, ~23 km cells)
PAD_NX <- 50
PAD_NY <- 50
PAD_NY        <- 50    # pad to 50 for transformer (15 rows of zeros at bottom)
N_BOOTSTRAPS  <- 1
N_SUBSAMPLE   <- 300   # out of ~349 stations
SEED          <- 2026
getwd()
setwd("C:/Users/nmorok/Documents/Thesis/Teleconnections_ViT/data/real/output")
OUTPUT_DIR    <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output"

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)


# ==============================================================================
# STEP 1: BUILD PREDICTION GRID
# ==============================================================================

build_grid <- function(survey_domain, station_locations_sf = NULL, cellsize = CELLSIZE) {
  #' Create irregular grid clipped to survey domain.
  #' Each valid cell knows its (row, col) position in the full regular grid.
  
  cat("[Grid] Building prediction grid...\n")
  
  # ── Full regular grid over bounding box ──
  full_grid <- st_make_grid(survey_domain, cellsize = cellsize,
                            crs = st_crs(survey_domain))
  
  # ── Which cells overlap the survey domain? ──
  hits <- st_intersects(full_grid, survey_domain, sparse = FALSE)[, 1]
  
  # ── Also force cells containing stations ──
  n_rescued <- 0
  if (!is.null(station_locations_sf)) {
    stn_hits <- st_intersects(full_grid, station_locations_sf, sparse = FALSE)
    has_station <- rowSums(stn_hits) > 0
    n_rescued <- sum(has_station & !hits)
    hits <- hits | has_station
    cat(sprintf("  Station forcing: %d cells rescued\n", n_rescued))
  }
  
  valid_grid <- full_grid[hits]
  valid_idx <- which(hits)
  
  # ── Grid dimensions ──
  bbox <- st_bbox(survey_domain)
  nx_full <- ceiling((bbox["xmax"] - bbox["xmin"]) / cellsize)
  ny_full <- ceiling((bbox["ymax"] - bbox["ymin"]) / cellsize)
  
  cat(sprintf("  Full grid: %d x %d = %d cells\n", nx_full, ny_full, nx_full * ny_full))
  cat(sprintf("  Valid cells: %d (%.1f%%)\n", length(valid_grid),
              100 * length(valid_grid) / (nx_full * ny_full)))
  
  # ── Map each valid cell to (row, col) in padded matrix ──
  # st_make_grid is column-major: index = (col-1)*ny + row, bottom-to-top
  grid_col <- ((valid_idx - 1) %% nx_full) + 1      # X varies fastest
  grid_row <- ((valid_idx - 1) %/% nx_full) + 1      # Y varies slowest (1 = south)
  grid_row <- ny_full + 1 - grid_row                  # flip: 1 = north
  
  # ── Centroids for SPDE projection ──
  centroids <- st_coordinates(st_centroid(valid_grid))
  
  # ── Verify station coverage ──
  if (!is.null(station_locations_sf)) {
    stn_check <- st_intersects(valid_grid, station_locations_sf, sparse = FALSE)
    n_covered <- sum(colSums(stn_check) > 0)
    n_total_stn <- ifelse("station" %in% names(station_locations_sf),
                          length(unique(station_locations_sf$station)),
                          nrow(station_locations_sf))
    cat(sprintf("  ✓ %d / %d stations covered by valid cells\n", n_covered, n_total_stn))
  }
  
  return(list(
    valid_grid = valid_grid,
    centroids  = centroids,
    grid_col   = grid_col,
    grid_row   = grid_row,
    nx_full    = nx_full,
    ny_full    = ny_full,
    n_valid    = length(valid_grid),
    cellsize   = cellsize,
    bbox       = bbox,
    valid_idx  = valid_idx
  ))
}


# ==============================================================================
# HELPER: Fill padded matrix from valid-cell values
# ==============================================================================

fill_matrix <- function(values, grid_info, pad_ny = PAD_NY, pad_nx = PAD_NX) {
  #' Place valid cell values into a padded matrix at their (row, col) positions.
  #' Cells outside the valid grid stay zero.
  
  mat <- matrix(0, nrow = pad_ny, ncol = pad_nx)
  n <- length(values)
  
  for (i in 1:n) {
    r <- grid_info$grid_row[i]
    c <- grid_info$grid_col[i]
    if (r >= 1 && r <= pad_ny && c >= 1 && c <= pad_nx) {
      mat[r, c] <- values[i]
    }
  }
  return(mat)
}


# ==============================================================================
# STEP 2: BUILD SPDE COMPONENTS
# ==============================================================================

build_spde <- function(station_coords, grid_info) {
  #' Build SPDE mesh and projection matrix to valid grid centroids.
  
  cat("[SPDE] Building mesh...\n")
  
  mesh <- fm_mesh_2d(
    loc = station_coords,
    cutoff = 30,
    max.edge = c(40, 100),
    offset = c(50, 150)
  )
  cat(sprintf("  Mesh: %d vertices, %d triangles\n", mesh$n, nrow(mesh$graph$tv)))
  
  spde <- inla.spde2.matern(mesh, alpha = 2)
  
  # Projection to valid grid centroids only
  A_grid <- inla.spde.make.A(mesh, loc = grid_info$centroids)
  
  return(list(mesh = mesh, spde = spde, A_grid = A_grid))
}


# ==============================================================================
# STEP 3: FIT SPDE FOR ONE YEAR
# ==============================================================================

fit_spde_year <- function(y, station_coords, mesh, spde, A_grid) {
  #' Fit spatial INLA model in log space and predict at grid centroids.
  #' Returns: vector [n_valid] of predicted density (original scale).
  
  y_log <- log1p(y)
  A_obs <- inla.spde.make.A(mesh, loc = station_coords)
  
  stack_obs <- inla.stack(
    data = list(y = y_log),
    A = list(A_obs, 1),
    effects = list(spatial = 1:spde$n.spde,
                   intercept = rep(1, length(y_log))),
    tag = "obs"
  )
  
  stack_pred <- inla.stack(
    data = list(y = NA),
    A = list(A_grid, 1),
    effects = list(spatial = 1:spde$n.spde,
                   intercept = rep(1, nrow(A_grid))),
    tag = "pred"
  )
  
  stack_full <- inla.stack(stack_obs, stack_pred)
  
  result <- tryCatch({
    inla(
      y ~ -1 + intercept + f(spatial, model = spde),
      data = inla.stack.data(stack_full),
      control.predictor = list(A = inla.stack.A(stack_full), compute = TRUE),
      control.compute = list(config = FALSE),
      control.inla = list(strategy = "gaussian", int.strategy = "eb"),
      verbose = FALSE
    )
  }, error = function(e) {
    warning(sprintf("INLA failed: %s", e$message))
    return(NULL)
  })
  
  if (is.null(result)) return(rep(0, nrow(A_grid)))
  
  idx_pred <- inla.stack.index(stack_full, tag = "pred")$data
  pred_log <- result$summary.fitted.values[idx_pred, "mean"]
  pred <- expm1(pred_log)
  pred[pred < 0] <- 0
  pred[is.na(pred)] <- 0
  
  return(pred)
}


# ==============================================================================
# STEP 4: BOOTSTRAP LOOP
# ==============================================================================

run_bootstrap <- function(data_sf, station_locations_sf, grid_info, spde_comps,
                          n_bootstraps = N_BOOTSTRAPS, n_subsample = N_SUBSAMPLE,
                          seed = SEED) {
  
  set.seed(seed)
  
  years <- sort(unique(data_sf$year))
  n_years <- length(years)
  all_stations <- unique(data_sf$station)
  n_stations <- length(all_stations)
  
  stn_coords_all <- st_coordinates(station_locations_sf)
  stn_names_all <- station_locations_sf$station
  
  cat(sprintf("\n  Bootstrap: %d stations, %d years, %d bootstraps\n",
              n_stations, n_years, n_bootstraps))
  cat(sprintf("  Subsampling %d / %d stations\n", n_subsample, n_stations))
  
  # Output: padded [n_bootstraps, n_years, 50, 50]
  output <- array(0, dim = c(n_bootstraps, n_years, PAD_NY, PAD_NX))
  
  total_fits <- n_bootstraps * n_years
  fit_count <- 0
  start_time <- Sys.time()
  
  for (b in 1:n_bootstraps) {
    sub_idx <- sample(1:n_stations, min(n_subsample, n_stations), replace = FALSE)
    sub_stations <- all_stations[sub_idx]
    
    for (y_idx in 1:n_years) {
      yr <- years[y_idx]
      
      yr_data <- data_sf %>% filter(year == yr, station %in% sub_stations)
      
      if (nrow(yr_data) == 0) {
        fit_count <- fit_count + 1
        next
      }
      
      avail_stations <- yr_data$station
      avail_coords <- stn_coords_all[match(avail_stations, stn_names_all), , drop = FALSE]
      
      # Fit SPDE → vector of length n_valid
      pred_valid <- fit_spde_year(
        y = yr_data$avg_dens,
        station_coords = avail_coords,
        mesh = spde_comps$mesh,
        spde = spde_comps$spde,
        A_grid = spde_comps$A_grid
      )
      
      # Fill into padded matrix
      output[b, y_idx, , ] <- fill_matrix(pred_valid, grid_info)
      
      fit_count <- fit_count + 1
      if (fit_count %% 50 == 0) {
        elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
        rate <- fit_count / elapsed
        remaining <- (total_fits - fit_count) / rate
        cat(sprintf("  [%d/%d] %.1f fits/min, ~%.0f min remaining\n",
                    fit_count, total_fits, rate, remaining))
      }
    }
    
    if (b %% 10 == 0 || b == 1) {
      cat(sprintf("  Bootstrap %d/%d complete\n", b, n_bootstraps))
    }
  }
  
  elapsed_total <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  cat(sprintf("  Done! %d fits in %.1f min (%.1f fits/min)\n",
              total_fits, elapsed_total, total_fits / elapsed_total))
  
  return(list(data = output, years = years))
}


# ==============================================================================
# STEP 5: SAVE OUTPUTS
# ==============================================================================

save_outputs <- function(spawner_result, recruit_result, grid_info,
                         output_dir = OUTPUT_DIR) {
  
  # Spatial mask
  mask <- fill_matrix(rep(1, grid_info$n_valid), grid_info)
  test_field <- spawner_result$data[1, 15, , ]  # bootstrap 1, year 15, [50, 50]
  
  
  # Save as RDS
  saveRDS(spawner_result$data, file.path(output_dir, "gridded_spawners.rds"))
  saveRDS(recruit_result$data, file.path(output_dir, "gridded_recruits.rds"))
  saveRDS(mask, file.path(output_dir, "spatial_mask.rds"))
  
  # Save mask as CSV too
  write.csv(mask, file.path(output_dir, "spatial_mask.csv"), row.names = FALSE)
  
  # Try .npy via reticulate
  if (requireNamespace("reticulate", quietly = TRUE)) {
    np <- reticulate::import("numpy")
    np$save(file.path(output_dir, "gridded_spawners.npy"), spawner_result$data)
    np$save(file.path(output_dir, "gridded_recruits.npy"), recruit_result$data)
    np$save(file.path(output_dir, "spatial_mask.npy"), mask)
    cat("Saved .npy files\n")
  }
  
  # Metadata
  metadata <- list(
    cellsize_km = grid_info$cellsize,
    nx_full = grid_info$nx_full,
    ny_full = grid_info$ny_full,
    pad_nx = PAD_NX,
    pad_ny = PAD_NY,
    n_valid_cells = grid_info$n_valid,
    n_bootstraps = dim(spawner_result$data)[1],
    spawner_years = spawner_result$years,
    recruit_years = recruit_result$years,
    n_spawner_years = length(spawner_result$years),
    n_recruit_years = length(recruit_result$years),
    crs = "+proj=utm +zone=2 +datum=WGS84 +units=km",
    bbox_xmin = as.numeric(grid_info$bbox["xmin"]),
    bbox_xmax = as.numeric(grid_info$bbox["xmax"]),
    bbox_ymin = as.numeric(grid_info$bbox["ymin"]),
    bbox_ymax = as.numeric(grid_info$bbox["ymax"])
  )
  
  write_json(metadata, file.path(output_dir, "grid_metadata.json"),
             pretty = TRUE, auto_unbox = TRUE)
  
  cat(sprintf("All outputs saved to %s/\n", output_dir))
  cat(sprintf("  Mask: %d valid cells in %dx%d grid\n", sum(mask), PAD_NY, PAD_NX))
}


# ==============================================================================
# STEP 6: VISUALIZATION
# ==============================================================================

plot_gridded_field <- function(pred_valid, grid_info, survey_domain, sf_maps,
                               title = "SPDE Gridded Field") {
  plot_sf <- st_sf(
    geometry = grid_info$valid_grid,
    val = pred_valid
  )
  # Don't hide zeros — let them show as dark on the color scale
  
  ggplot() +
    geom_sf(data = plot_sf, aes(fill = log1p(val)), color = NA) +
    geom_sf(data = survey_domain, fill = NA, color = "red", linewidth = 1) +
    geom_sf(data = sf_maps, fill = "grey70", color = "grey50") +
    scale_fill_viridis_c(name = "log1p(density)") +
    theme_minimal() +
    labs(title = title)
}


plot_year_comparison <- function(spawner_result, recruit_result, grid_info,
                                 survey_domain, sf_maps, yr_idx = NULL) {
  #' Plot mean field and a random bootstrap for one year.
  
  n_boot <- dim(spawner_result$data)[1]
  n_year <- dim(spawner_result$data)[2]
  if (is.null(yr_idx)) yr_idx <- ceiling(n_year / 2)
  target_year <- spawner_result$years[yr_idx]
  
  # Helper: extract valid-cell values from padded matrix
  extract_valid <- function(mat, grid_info) {
    vals <- numeric(grid_info$n_valid)
    for (i in 1:grid_info$n_valid) {
      vals[i] <- mat[grid_info$grid_row[i], grid_info$grid_col[i]]
    }
    return(vals)
  }
  
  # Mean across bootstraps
  mean_s <- apply(spawner_result$data[, yr_idx, , , drop = FALSE], c(3, 4), mean)
  mean_r <- apply(recruit_result$data[, yr_idx, , , drop = FALSE], c(3, 4), mean)
  
  # Random bootstrap
  set.seed(42)
  rand_b <- sample(1:n_boot, 1)
  rand_s <- spawner_result$data[rand_b, yr_idx, , ]
  rand_r <- recruit_result$data[rand_b, yr_idx, , ]
  
  library(patchwork)
  
  p1 <- plot_gridded_field(extract_valid(mean_s, grid_info), grid_info,
                           survey_domain, sf_maps,
                           sprintf("Spawner Mean — Year %d", target_year))
  p2 <- plot_gridded_field(extract_valid(rand_s, grid_info), grid_info,
                           survey_domain, sf_maps,
                           sprintf("Spawner Boot %d — Year %d", rand_b, target_year))
  p3 <- plot_gridded_field(extract_valid(mean_r, grid_info), grid_info,
                           survey_domain, sf_maps,
                           sprintf("Recruit Mean — Year %d", target_year))
  p4 <- plot_gridded_field(extract_valid(rand_r, grid_info), grid_info,
                           survey_domain, sf_maps,
                           sprintf("Recruit Boot %d — Year %d", rand_b, target_year))
  
  combined <- (p1 + p2) / (p3 + p4)
  print(combined)
  
  ggsave(file.path(OUTPUT_DIR, "bootstrap_comparison.png"), combined,
         width = 14, height = 12, dpi = 150)
}


plot_mask_check <- function(grid_info, survey_domain, station_locations_sf, sf_maps) {
  #' Verify mask alignment: valid cells (blue), stations (black dots).
  
  mask_sf <- st_sf(
    geometry = grid_info$valid_grid,
    valid = 1
  )
  
  p <- ggplot() +
    geom_sf(data = mask_sf, fill = "steelblue", alpha = 0.3, color = "steelblue", lwd = 0.2) +
    geom_sf(data = survey_domain, fill = NA, color = "red", linewidth = 1.2) +
    geom_sf(data = sf_maps, fill = "grey70", color = "grey50") +
    geom_sf(data = station_locations_sf, color = "black", size = 0.8) +
    theme_minimal() +
    labs(title = sprintf("Spatial Mask: %d valid cells, %d km cellsize",
                         grid_info$n_valid, grid_info$cellsize))
  
  print(p)
  return(p)
}


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

run_gridding <- function(spawner_sf, recruit_sf, station_locations_sf,
                         survey_domain, sf_maps = NULL,
                         n_bootstraps = N_BOOTSTRAPS,
                         n_subsample = N_SUBSAMPLE,
                         cellsize = CELLSIZE) {
  
  cat("=== SPDE Bootstrap Gridding Pipeline ===\n\n")
  
  # Ensure station column name
  if (!"station" %in% names(station_locations_sf)) {
    station_locations_sf <- station_locations_sf %>% rename(station = GIS_STATION)
  }
  
  cat(sprintf("Spawner: %d-%d (%d years)\n",
              min(spawner_sf$year), max(spawner_sf$year),
              length(unique(spawner_sf$year))))
  cat(sprintf("Recruit: %d-%d (%d years)\n",
              min(recruit_sf$year), max(recruit_sf$year),
              length(unique(recruit_sf$year))))
  
  # ── Step 1: Grid ──
  cat("\n[1/5] Building grid...\n")
  grid_info <- build_grid(survey_domain, station_locations_sf, cellsize)
  
  # ── Step 2: SPDE mesh ──
  cat("\n[2/5] Building SPDE mesh...\n")
  stn_coords <- st_coordinates(station_locations_sf)
  spde_comps <- build_spde(stn_coords, grid_info)
  
  # ── Quick sanity check: single year ──
  cat("\n[Sanity] Fitting one test year...\n")
  test_yr <- sort(unique(spawner_sf$year))[15]
  test_data <- spawner_sf %>% filter(year == test_yr)
  test_coords <- stn_coords[match(test_data$station, station_locations_sf$station), , drop = FALSE]
  
  test_pred <- fit_spde_year(test_data$avg_dens, test_coords,
                             spde_comps$mesh, spde_comps$spde, spde_comps$A_grid)
  cat(sprintf("  Year %d: range [%.0f, %.0f], nonzero: %d/%d\n",
              test_yr, min(test_pred), max(test_pred),
              sum(test_pred > 0), length(test_pred)))
  
  if (!is.null(sf_maps)) {
    p <- plot_gridded_field(test_pred, grid_info, survey_domain, sf_maps,
                            sprintf("Sanity Check — Year %d", test_yr))
    print(p)
  }
  
  # ── Step 3: Grid spawners ──
  cat("\n[3/5] Gridding spawners...\n")
  spawner_result <- run_bootstrap(
    spawner_sf, station_locations_sf, grid_info, spde_comps,
    n_bootstraps = n_bootstraps, n_subsample = n_subsample, seed = SEED
  )
  
  # ── Step 4: Grid recruits ──
  cat("\n[4/5] Gridding recruits...\n")
  recruit_result <- run_bootstrap(
    recruit_sf, station_locations_sf, grid_info, spde_comps,
    n_bootstraps = n_bootstraps, n_subsample = n_subsample, seed = SEED + 1000
  )
  
  # ── Step 5: Save ──
  cat("\n[5/5] Saving...\n")
  save_outputs(spawner_result, recruit_result, grid_info)
  
  # ── Plots ──
  if (!is.null(sf_maps)) {
    plot_mask_check(grid_info, survey_domain, station_locations_sf, sf_maps)
    plot_year_comparison(spawner_result, recruit_result, grid_info,
                         survey_domain, sf_maps)
  }
  
  return(list(
    spawner = spawner_result,
    recruit = recruit_result,
    grid_info = grid_info,
    spde_comps = spde_comps
  ))
}


# ==============================================================================
# RUN
# ==============================================================================

# After sourcing pipeline.R:
#
station_locations_sf_named <- station_locations_sf %>%
 rename(station = GIS_STATION)
#
# # Quick test (1 bootstrap)
test <- run_gridding(
  spawner_sf, recruit_sf, station_locations_sf_named,
 survey_domain, sf_maps,
 n_bootstraps = 100, n_subsample = 300
)

test2 <- test

test2

all_years <- 1988:2023  # full 36-year sequence
year_2020_idx <- which(all_years == 2020)  # = 33 in R (1-indexed)

# Create new arrays with room for 2020
n_boot <- dim(test$spawner$data)[1]
n_years_new <- 36
spawner_grids_new <- array(0, dim = c(n_boot, n_years_new, 50, 50))
recruit_grids_new <- array(0, dim = c(n_boot, n_years_new, 50, 50))

# Fill in everything before 2020 (indices 1:32 = years 1988-2019)
spawner_grids_new[, 1:(year_2020_idx - 1), , ] <- test$spawner$data[, 1:(year_2020_idx - 1), , ]
recruit_grids_new[, 1:(year_2020_idx - 1), , ] <- test$recruit$data[, 1:(year_2020_idx - 1), , ]

# Index 33 stays as zeros (2020)

# Fill in everything after 2020 (indices 34:36 = years 2021-2023)
spawner_grids_new[, (year_2020_idx + 1):n_years_new, , ] <- test$spawner$data[, year_2020_idx:35, , ]
recruit_grids_new[, (year_2020_idx + 1):n_years_new, , ] <- test$recruit$data[, year_2020_idx:35, , ]

# Save a year mask: 1 for valid, 0 for 2020
year_mask <- rep(1, n_years_new)
year_mask[year_2020_idx] <- 0

saveRDS(spawner_grids_new, file.path(OUTPUT_DIR, "gridded_spawners.rds"))
saveRDS(recruit_grids_new, file.path(OUTPUT_DIR, "gridded_recruits.rds"))
saveRDS(mask, file.path(OUTPUT_DIR, "spatial_mask.rds"))
saveRDS(year_mask, file.path(OUTPUT_DIR, "year_mask.rds"))
saveRDS(1988:2023, file.path(OUTPUT_DIR, "years.rds"))

if (requireNamespace("reticulate", quietly = TRUE)) {
  np <- reticulate::import("numpy")
  np$save(file.path(OUTPUT_DIR, "gridded_spawners.npy"), spawner_grids_new)
  np$save(file.path(OUTPUT_DIR, "gridded_recruits.npy"), recruit_grids_new)
  np$save(file.path(OUTPUT_DIR, "spatial_mask.npy"), mask)
  np$save(file.path(OUTPUT_DIR, "year_mask.npy"), as.integer(year_mask))
  np$save(file.path(OUTPUT_DIR, "years.npy"), as.integer(1988:2023))
  cat("Saved .npy files\n")
}






mask <- fill_matrix(rep(1, test$grid_info$n_valid), test$grid_info)
test_field <- test_result$spawner$data[1, 15, , ]  # bootstrap 1, year 15, [50, 50]
write.csv(test_field, file.path(OUTPUT_DIR,"test_spawner_yr15.csv"), row.names = FALSE)
write.csv(mask, file.path(OUTPUT_DIR,"test_mask.csv"), row.names = FALSE)  # mask from fill_matrix

# Also print what it should look like
cat("R dimensions:", dim(test_field), "\n")
cat("R [1,1]:", test_field[1,1], "\n")
cat("R [1,50]:", test_field[1,50], "\n")
cat("R [50,1]:", test_field[50,1], "\n")
cat("R max location:", which(test_field == max(test_field), arr.ind=TRUE), "\n")



test_field_vec <- test_result$spawner$data[1, 15, , ]
mask <- fill_matrix(rep(1, test$grid_info$n_valid), test$grid_info)






