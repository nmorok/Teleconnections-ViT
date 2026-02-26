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

GRID_NX       <- 50    # columns (E-W, ~23 km cells)
GRID_NY       <- 35    # rows (N-S, ~23 km cells)
PAD_NY        <- 50    # pad to 50 for transformer (15 rows of zeros at bottom)
N_BOOTSTRAPS  <- 1
N_SUBSAMPLE   <- 300   # out of ~349 stations
SEED          <- 2026
getwd()
setwd("C:/Users/nmorok/Documents/Thesis/Teleconnections_ViT/data/real/output")
OUTPUT_DIR    <- "C:/Users/nmorok/Documents/Thesis/Teleconnections_ViT/data/real/output"

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)


# ==============================================================================
# STEP 1: BUILD PREDICTION GRID
# ==============================================================================

build_prediction_grid <- function(survey_domain, nx = GRID_NX, ny = GRID_NY,
                                  station_locations_sf = NULL) {
  #' Create a regular grid covering the survey domain.
  #' 
  #' Mask is determined by polygon intersection (cell overlaps survey domain)
  
  #' plus station forcing (any cell containing a station is always valid).
  #' This prevents edge cells with stations from being incorrectly masked out.
  #'
  #' Args:
  #'   survey_domain: sf polygon defining the survey boundary
  #'   nx, ny: grid dimensions
  #'   station_locations_sf: sf data frame with station geometries (optional)
  #'     If provided, forces cells containing stations to be valid.
  #'
  #' Returns:
  #'   list with grid coordinates, mask matrix, bbox, cell sizes
  
  bbox <- st_bbox(survey_domain)
  
  # Cell edges
  x_edges <- seq(bbox["xmin"], bbox["xmax"], length.out = nx + 1)
  y_edges <- seq(bbox["ymin"], bbox["ymax"], length.out = ny + 1)
  
  # Cell centers (for SPDE projection)
  x_centers <- (x_edges[-length(x_edges)] + x_edges[-1]) / 2
  y_centers <- (y_edges[-length(y_edges)] + y_edges[-1]) / 2
  grid_coords <- expand.grid(X = x_centers, Y = y_centers)
  
  # ── Build grid cell polygons for proper intersection test ──
  # A cell is valid if ANY part of it overlaps the survey domain
  # (not just the center, which misses edge cells)
  cat("Building grid cell polygons for intersection test...\n")
  
  grid_polys <- vector("list", nx * ny)
  idx <- 1
  for (j in 1:ny) {
    for (i in 1:nx) {
      coords <- matrix(c(
        x_edges[i],   y_edges[j],
        x_edges[i+1], y_edges[j],
        x_edges[i+1], y_edges[j+1],
        x_edges[i],   y_edges[j+1],
        x_edges[i],   y_edges[j]
      ), ncol = 2, byrow = TRUE)
      grid_polys[[idx]] <- st_polygon(list(coords))
      idx <- idx + 1
    }
  }
  
  grid_sfc <- st_sfc(grid_polys, crs = st_crs(survey_domain))
  
  # Test which cell polygons intersect the survey domain
  inside <- st_intersects(grid_sfc, survey_domain, sparse = FALSE)[, 1]
  
  n_polygon_valid <- sum(inside)
  cat(sprintf("  Polygon intersection: %d cells overlap survey domain\n", n_polygon_valid))
  
  # ── Force cells containing stations to be valid ──
  # Prevents edge stations from falling in masked cells
  n_station_forced <- 0
  if (!is.null(station_locations_sf)) {
    stn_coords <- st_coordinates(station_locations_sf)
    
    stn_cell_x <- findInterval(stn_coords[, 1], x_edges)
    stn_cell_y <- findInterval(stn_coords[, 2], y_edges)
    stn_cell_x <- pmin(pmax(stn_cell_x, 1), nx)
    stn_cell_y <- pmin(pmax(stn_cell_y, 1), ny)
    
    # Linear index matching grid_polys order (column-major: j varies first within each i)
    stn_linear_idx <- (stn_cell_x - 1) * ny + stn_cell_y
    
    # How many stations were in masked cells?
    n_station_forced <- sum(!inside[unique(stn_linear_idx)])
    
    # Force valid
    inside[unique(stn_linear_idx)] <- TRUE
    
    cat(sprintf("  Station forcing: %d additional cells activated (%d stations rescued)\n",
                n_station_forced, n_station_forced))
  }
  
  # ── Reshape to mask matrix ──
  # grid_polys are ordered: j (Y) varies fastest, i (X) varies slowest
  # matrix() fills column-by-column, so nrow=ny, ncol=nx gives correct layout
  mask_matrix <- matrix(as.integer(inside), nrow = ny, ncol = nx, byrow = FALSE)
  mask_matrix <- mask_matrix[ny:1, ]  # flip so row 0 = north (top)
  
  # Cell size
  dx <- diff(x_edges[1:2])
  dy <- diff(y_edges[1:2])
  
  cat(sprintf("\nGrid: %d x %d = %d cells\n", nx, ny, nx * ny))
  cat(sprintf("Valid cells: %d (%.1f%%)\n", sum(mask_matrix), 
              100 * sum(mask_matrix) / (nx * ny) ))
  cat(sprintf("Cell size: %.1f x %.1f km\n", dx, dy))
  
  # ── Verify all stations are in valid cells ──
  if (!is.null(station_locations_sf)) {
    stn_coords <- st_coordinates(station_locations_sf)
    stn_cell_x <- findInterval(stn_coords[, 1], x_edges)
    stn_cell_y <- findInterval(stn_coords[, 2], y_edges)
    stn_cell_x <- pmin(pmax(stn_cell_x, 1), nx)
    stn_cell_y <- pmin(pmax(stn_cell_y, 1), ny)
    
    row_idx <- ny + 1 - stn_cell_y
    col_idx <- stn_cell_x
    mask_vals <- mask_matrix[cbind(row_idx, col_idx)]
    n_outside <- sum(mask_vals == 0)
    
    if (n_outside > 0) {
      warning(sprintf("%d stations still outside mask after forcing!", n_outside))
    } else {
      cat(sprintf("✓ All %d stations are inside valid cells\n", 
                  nrow(stn_coords)))
    }
  }
  
  return(list(
    x_centers = x_centers,
    y_centers = y_centers,
    x_edges = x_edges,
    y_edges = y_edges,
    grid_coords = as.matrix(grid_coords),   # [nx*ny, 2] for SPDE projection
    mask = mask_matrix,                       # [ny, nx] binary
    bbox = bbox,
    dx = dx, dy = dy,
    nx = nx, ny = ny
  ))
}


# ==============================================================================
# STEP 2: BUILD MESH AND PROJECTION MATRICES
# ==============================================================================

build_spde_components <- function(station_coords, grid_info) {
  #' Build SPDE mesh from station locations and create projection matrices.
  #'
  #' station_coords: matrix [n_stations, 2] in UTM km
  #' grid_info: output from build_prediction_grid()
  
  # Build mesh (same pattern as preprocess_obj in EOF_GLLVM_functions.R)
  mesh <- fm_mesh_2d(
    loc = station_coords,
    cutoff = 30,         # minimum distance between mesh nodes (km)
    max.edge = c(40, 100), # max triangle edge: inner, outer (km)
    offset = c(50, 150)   # inner/outer buffer (km)
  )
  
  cat(sprintf("Mesh: %d vertices, %d triangles\n", mesh$n, nrow(mesh$graph$tv)))
  
  # SPDE model
  spde <- inla.spde2.matern(mesh, alpha = 2)
  
  # Projection matrix: grid points
  A_grid <- inla.spde.make.A(mesh, loc = grid_info$grid_coords)
  
  return(list(
    mesh = mesh,
    spde = spde,
    A_grid = A_grid
  ))
}


# ==============================================================================
# STEP 3: FIT SPDE MODEL FOR ONE YEAR/SUBSAMPLE
# ==============================================================================

fit_spde_year <- function(y, station_coords, mesh, spde, A_grid, 
                          use_log = TRUE) {
  #' Fit a simple spatial INLA model and project to grid.
  #'
  #' Args:
  #'   y: vector of density values at stations
  #'   station_coords: matrix [n_stations, 2]
  #'   mesh, spde: from build_spde_components()
  #'   A_grid: projection matrix to grid
  #'   use_log: if TRUE, model log(1 + y) then back-transform
  #'
  #' Returns:
  #'   grid_pred: vector [nx * ny] of predicted density at grid cells
  
  # Transform response
  if (use_log) {
    y_model <- log1p(y)
  } else {
    y_model <- y
  }
  
  # Observation projection matrix (station locations → mesh)
  A_obs <- inla.spde.make.A(mesh, loc = station_coords)
  
  # INLA stack
  stack_obs <- inla.stack(
    data = list(y = y_model),
    A = list(A_obs, 1),
    effects = list(
      spatial = 1:spde$n.spde,
      intercept = rep(1, length(y_model))
    ),
    tag = "obs"
  )
  
  # Grid prediction stack (NA response = predict)
  stack_pred <- inla.stack(
    data = list(y = NA),
    A = list(A_grid, 1),
    effects = list(
      spatial = 1:spde$n.spde,
      intercept = rep(1, nrow(A_grid))
    ),
    tag = "pred"
  )
  
  # Combine stacks
  stack_full <- inla.stack(stack_obs, stack_pred)
  
  # Fit model (fast settings)
  formula <- y ~ -1 + intercept + f(spatial, model = spde)
  
  result <- tryCatch({
    inla(
      formula,
      data = inla.stack.data(stack_full),
      control.predictor = list(
        A = inla.stack.A(stack_full),
        compute = TRUE
      ),
      control.compute = list(config = FALSE),
      control.inla = list(
        strategy = "gaussian",    # fastest approximation
        int.strategy = "eb"       # empirical bayes (no integration)
      ),
      verbose = FALSE
    )
  }, error = function(e) {
    warning(sprintf("INLA fit failed: %s", e$message))
    return(NULL)
  })
  
  if (is.null(result)) {
    return(rep(NA, nrow(A_grid)))
  }
  
  # Extract predictions at grid locations
  idx_pred <- inla.stack.index(stack_full, tag = "pred")$data
  grid_pred <- result$summary.fitted.values[idx_pred, "mean"]
  
  # Back-transform
  if (use_log) {
    grid_pred <- expm1(grid_pred)
    grid_pred[grid_pred < 0] <- 0
  }
  
  return(grid_pred)
}


# ==============================================================================
# STEP 4: BOOTSTRAP LOOP
# ==============================================================================

run_bootstrap_gridding <- function(data_sf, station_locations_sf, 
                                   survey_domain, grid_info, spde_comps,
                                   n_bootstraps = N_BOOTSTRAPS,
                                   n_subsample = N_SUBSAMPLE,
                                   seed = SEED) {
  #' Run the full bootstrap gridding pipeline for one variable (spawner or recruit).
  #'
  #' Args:
  #'   data_sf: sf data frame with columns: station, year, avg_dens, geometry
  #'   station_locations_sf: sf data frame with all station locations
  #'   survey_domain: sf polygon
  #'   grid_info: from build_prediction_grid()
  #'   spde_comps: from build_spde_components()
  #'
  #' Returns:
  #'   array [n_bootstraps, n_years, PAD_NY, GRID_NX] — padded to 50x50
  
  set.seed(seed)
  
  years <- sort(unique(data_sf$year))
  n_years <- length(years)
  all_stations <- unique(data_sf$station)
  n_stations <- length(all_stations)
  
  cat(sprintf("\nBootstrap gridding: %d stations, %d years, %d bootstraps\n",
              n_stations, n_years, n_bootstraps))
  cat(sprintf("Subsampling %d / %d stations per bootstrap\n", 
              n_subsample, n_stations))
  
  # Pre-allocate output (padded to 50x50)
  output <- array(0, dim = c(n_bootstraps, n_years, PAD_NY, GRID_NX))
  
  # Get all station coordinates (for subsetting later)
  station_coords_all <- st_coordinates(station_locations_sf)
  station_names_all <- station_locations_sf$station
  
  # Progress tracking
  total_fits <- n_bootstraps * n_years
  fit_count <- 0
  start_time <- Sys.time()
  
  for (b in 1:n_bootstraps) {
    # Subsample stations (same subset for all years in this bootstrap)
    sub_idx <- sample(1:n_stations, n_subsample, replace = FALSE)
    sub_stations <- all_stations[sub_idx]
    sub_coords <- station_coords_all[match(sub_stations, station_names_all), , drop = FALSE]
    
    for (y_idx in 1:n_years) {
      yr <- years[y_idx]
      
      # Get data for this year and subsampled stations
      yr_data <- data_sf %>%
        filter(year == yr, station %in% sub_stations)
      
      # Some stations may not have data this year — use what's available
      available_stations <- yr_data$station
      available_coords <- station_coords_all[match(available_stations, station_names_all), , drop = FALSE]
      
      # Fit SPDE and predict to grid
      grid_pred <- fit_spde_year(
        y = yr_data$avg_dens,
        station_coords = available_coords,
        mesh = spde_comps$mesh,
        spde = spde_comps$spde,
        A_grid = spde_comps$A_grid,
        use_log = TRUE
      )
      
      # Reshape to [ny, nx] and apply mask
      pred_matrix <- matrix(grid_pred, nrow = GRID_NY, ncol = GRID_NX, byrow = FALSE)
      pred_matrix <- pred_matrix[GRID_NY:1, ]   # flip to row 0 = north
      pred_matrix[grid_info$mask == 0] <- 0       # zero outside survey area
      pred_matrix[is.na(pred_matrix)] <- 0        # handle any NAs
      
      # Store in padded array (real data in rows 1:35, rows 36:50 stay zero)
      output[b, y_idx, 1:GRID_NY, ] <- pred_matrix
      
      fit_count <- fit_count + 1
      if (fit_count %% 50 == 0) {
        elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
        rate <- fit_count / elapsed
        remaining <- (total_fits - fit_count) / rate
        cat(sprintf("  [%d/%d] %.0f fits/min, ~%.0f min remaining\n",
                    fit_count, total_fits, rate, remaining))
      }
    }
    
    if (b %% 10 == 0) {
      cat(sprintf("  Bootstrap %d/%d complete\n", b, n_bootstraps))
    }
  }
  
  elapsed_total <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  cat(sprintf("\nDone! %d fits in %.1f minutes (%.1f fits/min)\n",
              total_fits, elapsed_total, total_fits / elapsed_total))
  
  return(list(
    data = output,
    years = years
  ))
}


# ==============================================================================
# STEP 5: SAVE OUTPUTS
# ==============================================================================

save_outputs <- function(spawner_result, recruit_result, grid_info, output_dir = OUTPUT_DIR) {
  #' Save gridded arrays and metadata.
  #' 
  #' Uses reticulate to save .npy files (numpy format).
  #' Falls back to .rds if reticulate is not available.
  
  # --- Spatial mask (padded to 50x50) ---
  mask_padded <- matrix(0, nrow = PAD_NY, ncol = GRID_NX)
  mask_padded[1:GRID_NY, ] <- grid_info$mask
  
  # --- Try saving as .npy via reticulate ---
  use_npy <- requireNamespace("reticulate", quietly = TRUE)
  
  if (use_npy) {
    library(reticulate)
    np <- import("numpy")
    
    np$save(file.path(output_dir, "gridded_spawners.npy"), spawner_result$data)
    np$save(file.path(output_dir, "gridded_recruits.npy"), recruit_result$data)
    np$save(file.path(output_dir, "spatial_mask.npy"), mask_padded)
    
    cat("Saved .npy files via reticulate\n")
  } 
  
  # Fallback: save as RDS (load in Python with rpy2 or convert manually)
  saveRDS(spawner_result$data, file.path(output_dir, "gridded_spawners.rds"))
  saveRDS(recruit_result$data, file.path(output_dir, "gridded_recruits.rds"))
  saveRDS(mask_padded, file.path(output_dir, "spatial_mask.rds"))
    
  # Also save as CSV for the mask (it's small)
  write.csv(mask_padded, file.path(output_dir, "spatial_mask.csv"), row.names = FALSE)
    
  cat("Saved .rds files\n")

  
  
  # --- Metadata (always save as JSON) ---
  metadata <- list(
    grid_nx = GRID_NX,
    grid_ny = GRID_NY,
    pad_ny = PAD_NY,
    n_bootstraps = N_BOOTSTRAPS,
    n_subsample = N_SUBSAMPLE,
    cell_dx_km = grid_info$dx,
    cell_dy_km = grid_info$dy,
    bbox_xmin = as.numeric(grid_info$bbox["xmin"]),
    bbox_xmax = as.numeric(grid_info$bbox["xmax"]),
    bbox_ymin = as.numeric(grid_info$bbox["ymin"]),
    bbox_ymax = as.numeric(grid_info$bbox["ymax"]),
    spawner_years = spawner_result$years,
    recruit_years = recruit_result$years,
    n_spawner_years = length(spawner_result$years),
    n_recruit_years = length(recruit_result$years),
    crs = "+proj=utm +zone=2 +datum=WGS84 +units=km",
    real_data_rows = paste0("1:", GRID_NY),
    padding_rows = paste0((GRID_NY + 1), ":", PAD_NY)
  )
  
  write_json(metadata, file.path(output_dir, "grid_metadata.json"), 
             pretty = TRUE, auto_unbox = TRUE)
  
  cat(sprintf("\nAll outputs saved to %s/\n", output_dir))
}


# ==============================================================================
# STEP 6: VISUALIZATION (sanity check)
# ==============================================================================

plot_grid_check <- function(spawner_result, recruit_result, grid_info,
                            survey_domain, sf_maps, output_dir = OUTPUT_DIR) {
  #' Quick sanity plots: mask, sample year, mean field.
  
  library(ggplot2)
  
  # --- Mask visualization ---
  mask_df <- expand.grid(
    col = 1:GRID_NX,
    row = 1:GRID_NY
  )
  mask_df$valid <- as.vector(grid_info$mask)
  
  p_mask <- ggplot(mask_df, aes(x = col, y = row, fill = factor(valid))) +
    geom_tile() +
    scale_fill_manual(values = c("0" = "grey90", "1" = "steelblue"), 
                      name = "Valid") +
    coord_equal() +
    labs(title = sprintf("Spatial Mask: %d/%d cells (%.0f%%)",
                         sum(grid_info$mask), length(grid_info$mask),
                         100 * mean(grid_info$mask))) +
    theme_minimal()
  
  ggsave(file.path(output_dir, "grid_mask_check.png"), p_mask, 
         width = 10, height = 7, dpi = 150)
  
  # --- Sample gridded field (bootstrap 1, middle year) ---
  mid_yr <- ceiling(dim(spawner_result$data)[2] / 2)
  
  field_s <- spawner_result$data[1, mid_yr, 1:GRID_NY, ]
  field_r <- recruit_result$data[1, mid_yr, 1:GRID_NY, ]
  
  field_df <- expand.grid(col = 1:GRID_NX, row = 1:GRID_NY)
  field_df$spawner <- log1p(as.vector(field_s))
  field_df$recruit <- log1p(as.vector(field_r))
  
  p_fields <- cowplot::plot_grid(
    ggplot(field_df, aes(col, row, fill = spawner)) +
      geom_tile() + scale_fill_viridis_c() + coord_equal() +
      labs(title = sprintf("Spawner (B1, Yr %d) [log]", mid_yr)) +
      theme_minimal(),
    ggplot(field_df, aes(col, row, fill = recruit)) +
      geom_tile() + scale_fill_viridis_c(option = "plasma") + coord_equal() +
      labs(title = sprintf("Recruit (B1, Yr %d) [log]", mid_yr)) +
      theme_minimal(),
    ncol = 2
  )
  field_df$x_utm <- grid_info$grid_coords[,1]
  field_df$y_utm <- grid_info$grid_coords[,2]
  
  ggplot(field_df, aes(x = x_utm, y = y_utm, fill = spawner)) +
    geom_tile() +
    geom_sf(data = survey_domain, inherit.aes = FALSE, fill = NA, color = "red") +
    coord_sf() # This ensures the map projection is respected
  
  ggsave(file.path(output_dir, "grid_field_check.png"), p_fields, 
         width = 14, height = 6, dpi = 150)
  
  cat("Sanity check plots saved\n")
}


plot_grid_check <- function(spawner_result, recruit_result, grid_info,
                            survey_domain, sf_maps, output_dir = OUTPUT_DIR) {
  #' Quick sanity plots: mask, sample year, mean field.
  
  library(ggplot2)
  
  # --- Mask visualization ---
  mask_df <- expand.grid(
    col = 1:GRID_NX,
    row = 1:GRID_NY
  )
  mask_df$valid <- as.vector(grid_info$mask)
  
  p_mask <- ggplot(mask_df, aes(x = col, y = row, fill = factor(valid))) +
    geom_tile() +
    scale_fill_manual(values = c("0" = "grey90", "1" = "steelblue"), 
                      name = "Valid") +
    coord_equal() +
    labs(title = sprintf("Spatial Mask: %d/%d cells (%.0f%%)",
                         sum(grid_info$mask), length(grid_info$mask),
                         100 * mean(grid_info$mask))) +
    theme_minimal()
  
  ggsave(file.path(output_dir, "grid_mask_check.png"), p_mask, 
         width = 10, height = 7, dpi = 150)
  
  # --- Sample gridded field (bootstrap 1, middle year) ---
  mid_yr <- ceiling(dim(spawner_result$data)[2] / 2)
  
  field_s <- spawner_result$data[1, mid_yr, 1:GRID_NY, ]
  field_r <- recruit_result$data[1, mid_yr, 1:GRID_NY, ]
  
  field_df <- expand.grid(col = 1:GRID_NX, row = 1:GRID_NY)
  field_df$spawner <- log1p(as.vector(field_s))
  field_df$recruit <- log1p(as.vector(field_r))
  
  p_fields <- cowplot::plot_grid(
    ggplot(field_df, aes(col, row, fill = spawner)) +
      geom_tile() + scale_fill_viridis_c() + coord_equal() +
      labs(title = sprintf("Spawner (B1, Yr %d) [log]", mid_yr)) +
      theme_minimal(),
    ggplot(field_df, aes(col, row, fill = recruit)) +
      geom_tile() + scale_fill_viridis_c(option = "plasma") + coord_equal() +
      labs(title = sprintf("Recruit (B1, Yr %d) [log]", mid_yr)) +
      theme_minimal(),
    ncol = 2
  )
  
  ggsave(file.path(output_dir, "grid_field_check.png"), p_fields, 
         width = 14, height = 6, dpi = 150)
  
  cat("Sanity check plots saved\n")
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
#'
#' Run this AFTER pipeline.R has created:
#'   - spawner_sf, recruit_sf (sf data frames)
#'   - station_locations_sf (sf data frame with station geometry)
#'   - survey_domain (sf polygon)
#'   - sf_maps (coastline for plotting)
#'

run_gridding <- function(spawner_sf, recruit_sf, station_locations_sf,
                         survey_domain, sf_maps = NULL,
                         n_bootstraps = N_BOOTSTRAPS, 
                         n_subsample = N_SUBSAMPLE,
                         exclude_years = c(2015, 2020)) {
  
  cat("="," SPDE Bootstrap Gridding Pipeline ", "=", "\n\n")
  
  # Ensure station_locations_sf has a 'station' column
  if (!"station" %in% names(station_locations_sf)) {
    station_locations_sf <- station_locations_sf %>% 
      rename(station = GIS_STATION)
  }
  
  # Filter out excluded years (2020 = COVID)
  #spawner_sf <- spawner_sf %>% filter(!year %in% c(2015))
  #recruit_sf <- recruit_sf %>% filter(!year %in% )
  
  cat(sprintf("Spawner years: %d-%d (%d years)\n",
              min(spawner_sf$year), max(spawner_sf$year),
              length(unique(spawner_sf$year))))
  cat(sprintf("Recruit years: %d-%d (%d years)\n",
              min(recruit_sf$year), max(recruit_sf$year),
              length(unique(recruit_sf$year))))
  
  # --- Step 1: Build grid ---
  cat("\n[1/5] Building prediction grid...\n")
  grid_info <- build_prediction_grid(survey_domain, nx = GRID_NX, ny = GRID_NY,
                                     station_locations_sf = station_locations_sf)
  
  # --- Step 2: Build SPDE mesh ---
  cat("\n[2/5] Building SPDE mesh...\n")
  station_coords <- st_coordinates(station_locations_sf)
  spde_comps <- build_spde_components(station_coords, grid_info)
  
  # --- Step 3: Grid spawners ---
  cat("\n[3/5] Gridding spawner data...\n")
  spawner_result <- run_bootstrap_gridding(
    data_sf = spawner_sf,
    station_locations_sf = station_locations_sf,
    survey_domain = survey_domain,
    grid_info = grid_info,
    spde_comps = spde_comps,
    n_bootstraps = n_bootstraps,
    n_subsample = n_subsample,
    seed = SEED
  )
  
  # --- Step 4: Grid recruits ---
  cat("\n[4/5] Gridding recruit data...\n")
  recruit_result <- run_bootstrap_gridding(
    data_sf = recruit_sf,
    station_locations_sf = station_locations_sf,
    survey_domain = survey_domain,
    grid_info = grid_info,
    spde_comps = spde_comps,
    n_bootstraps = n_bootstraps,
    n_subsample = n_subsample,
    seed = SEED + 1000  # different seed for recruit subsamples
  )
  
  # --- Step 5: Save ---
  cat("\n[5/5] Saving outputs...\n")
  save_outputs(spawner_result, recruit_result, grid_info)
  
  # --- Sanity checks ---
  if (!is.null(sf_maps)) {
    plot_grid_check(spawner_result, recruit_result, grid_info,
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
# QUICK TEST MODE (5 bootstraps, for verifying pipeline works)
# ==============================================================================

# Uncomment to run after sourcing pipeline.R:
#
# # Rename station column if needed
station_locations_sf_named <- station_locations_sf %>%
  rename(station = GIS_STATION)
#
# # Quick test with 5 bootstraps
test_result <- run_gridding(
  spawner_sf = spawner_sf,
  recruit_sf = recruit_sf,
  station_locations_sf = station_locations_sf_named,
  survey_domain = survey_domain,
  sf_maps = sf_maps,
  n_bootstraps = 1,
  n_subsample = 300
)
#
# # Full run
# full_result <- run_gridding(
#   spawner_sf = spawner_sf,
#   recruit_sf = recruit_sf,
#   station_locations_sf = station_locations_sf_named,
#   survey_domain = survey_domain,
#   sf_maps = sf_maps,
#   n_bootstraps = 100,
#   n_subsample = 300
# )



library(ggplot2)
library(dplyr)
library(patchwork) # for combining plots

plot_bootstrap_comparison <- function(spawner_result, recruit_result, grid_info, yr_idx = NULL) {
  # 1. Setup dimensions
  n_boot <- dim(spawner_result$data)[1]
  n_year <- dim(spawner_result$data)[2]
  
  # Pick a middle year if not specified
  if(is.null(yr_idx)) yr_idx <- ceiling(n_year / 2)
  target_year <- spawner_result$years[yr_idx]
  
  # 2. Calculate Means and Pick Random Bootstrap
  set.seed(42) # for reproducibility of the "random" pick
  rand_b <- sample(1:n_boot, 1)
  
  # Mean across the first dimension (bootstraps)
  mean_s <- apply(spawner_result$data[, yr_idx, 1:GRID_NY, ], c(2, 3), mean)
  mean_r <- apply(recruit_result$data[, yr_idx, 1:GRID_NY, ], c(2, 3), mean)
  
  # Single bootstrap slice
  rand_s <- spawner_result$data[rand_b, yr_idx, 1:GRID_NY, ]
  rand_r <- recruit_result$data[rand_b, yr_idx, 1:GRID_NY, ]
  
  # 3. Helper to Convert Matrix to Long-Form DF for ggplot
  prep_df <- function(mat, name) {
    df <- expand.grid(col = 1:GRID_NX, row = 1:GRID_NY)
    df$val <- as.vector(mat)
    df$type <- name
    # Mask out zeroes for better visualization
    df$val[df$val == 0] <- NA 
    return(df)
  }
  
  # Combine into one big plotting DF
  plot_df <- rbind(
    prep_df(mean_s, "Spawner: Mean Field"),
    prep_df(rand_s, paste0("Spawner: Bootstrap ", rand_b)),
    prep_df(mean_r, "Recruit: Mean Field"),
    prep_df(rand_r, paste0("Recruit: Bootstrap ", rand_b))
  )
  
  # 4. Create the Plot
  p <- ggplot(plot_df, aes(x = col, y = row, fill = log1p(val))) +
    geom_tile() +
    facet_wrap(~type, ncol = 2) +
    scale_fill_viridis_c(option = "magma", na.value = "grey95") +
    coord_equal() +
    theme_minimal() +
    labs(
      title = paste("EBS Crab Distribution - Year:", target_year),
      subtitle = "Comparing Ensemble Mean vs. Single Stochastic Realization",
      fill = "log1p(Density)"
    )
  
  print(p)
  ggsave("bootstrap_comparison.png", p, width = 12, height = 10)
}

# Run it
plot_bootstrap_comparison(spawner_result, recruit_result, grid_info)