# ==============================================================================
# SPDE-Based Gridding of EBS Survey Data with Bootstrap Subsampling
# Tweedie p estimated per year from full station set, then fixed in bootstraps
# ==============================================================================

library(INLA)
library(fmesher)
library(sf)
library(dplyr)
library(jsonlite)
library(parallel)
library(R.utils)
library(ggplot2)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CELLSIZE     <- 25
PAD_NX       <- 50
PAD_NY       <- 50
N_BOOTSTRAPS <- 100
N_SUBSAMPLE  <- 300
SEED         <- 2026
OUTPUT_DIR   <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output/pub/Gaussian/300"

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)


# ==============================================================================
# STEP 1: BUILD PREDICTION GRID
# ==============================================================================

build_grid <- function(survey_domain, station_locations_sf = NULL, cellsize = CELLSIZE) {
  
  cat("[Grid] Building prediction grid...\n")
  
  full_grid <- st_make_grid(survey_domain, cellsize = cellsize,
                            crs = st_crs(survey_domain))
  
  hits <- st_intersects(full_grid, survey_domain, sparse = FALSE)[, 1]
  
  n_rescued <- 0
  if (!is.null(station_locations_sf)) {
    stn_hits    <- st_intersects(full_grid, station_locations_sf, sparse = FALSE)
    has_station <- rowSums(stn_hits) > 0
    n_rescued   <- sum(has_station & !hits)
    hits        <- hits | has_station
    cat(sprintf("  Station forcing: %d cells rescued\n", n_rescued))
  }
  
  valid_grid <- full_grid[hits]
  valid_idx  <- which(hits)
  
  bbox    <- st_bbox(survey_domain)
  nx_full <- ceiling((bbox["xmax"] - bbox["xmin"]) / cellsize)
  ny_full <- ceiling((bbox["ymax"] - bbox["ymin"]) / cellsize)
  
  cat(sprintf("  Full grid: %d x %d = %d cells\n", nx_full, ny_full, nx_full * ny_full))
  cat(sprintf("  Valid cells: %d (%.1f%%)\n", length(valid_grid),
              100 * length(valid_grid) / (nx_full * ny_full)))
  
  grid_col <- ((valid_idx - 1) %% nx_full) + 1
  grid_row <- ((valid_idx - 1) %/% nx_full) + 1
  grid_row <- ny_full + 1 - grid_row
  
  centroids <- st_coordinates(st_centroid(valid_grid))
  
  if (!is.null(station_locations_sf)) {
    stn_check   <- st_intersects(valid_grid, station_locations_sf, sparse = FALSE)
    n_covered   <- sum(colSums(stn_check) > 0)
    n_total_stn <- ifelse("station" %in% names(station_locations_sf),
                          length(unique(station_locations_sf$station)),
                          nrow(station_locations_sf))
    cat(sprintf("  \u2713 %d / %d stations covered by valid cells\n", n_covered, n_total_stn))
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
  
  mat <- matrix(0, nrow = pad_ny, ncol = pad_nx)
  n   <- length(values)
  
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
  
  cat("[SPDE] Building mesh...\n")
  
  mesh <- fm_mesh_2d(
    loc      = station_coords,
    cutoff   = 30,
    max.edge = c(40, 100),
    offset   = c(50, 150)
  )
  cat(sprintf("  Mesh: %d vertices, %d triangles\n", mesh$n, nrow(mesh$graph$tv)))
  
  spde   <- inla.spde2.matern(mesh, alpha = 2)
  A_grid <- inla.spde.make.A(mesh, loc = grid_info$centroids)
  
  return(list(mesh = mesh, spde = spde, A_grid = A_grid))
}


# ==============================================================================
# STEP 3: ESTIMATE TWEEDIE p PER YEAR FROM FULL STATION SET
# Fits one obs-only Tweedie INLA per year (no subsampling, no prediction stack)
# to obtain a stable p_t for each year t. These are then held fixed during
# bootstrap spatial interpolation, removing the joint p-estimation that causes
# convergence failures in degenerate subsamples (e.g. B022 / 2017).
# ==============================================================================

estimate_tweedie_p_by_year <- function(data_sf, station_locations_sf, spde_comps,
                                       timeout_sec = 1800,   # 30 minutes per year
                                       output_dir  = OUTPUT_DIR) {
  
  years          <- sort(unique(data_sf$year))
  p_by_year      <- setNames(rep(NA_real_, length(years)), as.character(years))
  stn_coords_all <- st_coordinates(station_locations_sf)
  stn_names_all  <- station_locations_sf$station
  
  cat(sprintf("[p-estimation] Estimating Tweedie p for %d years using full station set...\n",
              length(years)))
  cat(sprintf("  Timeout per year: %.0f minutes\n", timeout_sec / 60))
  
  for (yr in years) {
    
    yr_data <- data_sf %>% filter(year == yr)
    
    if (nrow(yr_data) == 0) {
      cat(sprintf("  Year %d: SKIPPED (no data)\n", yr))
      next
    }
    
    avail_coords <- stn_coords_all[
      match(yr_data$station, stn_names_all), , drop = FALSE
    ]
    y_scaled <- yr_data$avg_dens / 1000
    A_obs    <- inla.spde.make.A(spde_comps$mesh, loc = avail_coords)
    
    # Obs-only stack — no prediction grid needed, just hyperparameter estimation
    stack <- inla.stack(
      data    = list(y = y_scaled),
      A       = list(A_obs, 1),
      effects = list(spatial   = 1:spde_comps$spde$n.spde,
                     intercept = rep(1, length(y_scaled))),
      tag = "obs"
    )
    
    cat(sprintf("  Year %d: fitting (n_stations = %d)...\n", yr, nrow(yr_data)))
    t0 <- proc.time()["elapsed"]
    
    fit <- tryCatch({
      R.utils::withTimeout({
        inla(
          y ~ -1 + intercept + f(spatial, model = spde_comps$spde),
          family            = "tweedie",
          data              = inla.stack.data(stack),
          control.family    = list(link = "log"),
          control.predictor = list(A = inla.stack.A(stack), compute = FALSE),
          control.compute   = list(config = FALSE),
          control.inla      = list(strategy = "gaussian", int.strategy = "eb"),
          num.threads       = "1:1",
          verbose           = FALSE
        )
      }, timeout = timeout_sec, onTimeout = "error")
    }, error = function(e) e)   # capture error object, don't swallow it
    
    elapsed <- proc.time()["elapsed"] - t0
    
    if (inherits(fit, "error")) {
      # Hard stop — do not interpolate, do not continue
      stop(sprintf(
        "\n[p-estimation] FAILED for year %d after %.0fs.\n  Error: %s\n\n%s",
        yr, elapsed, conditionMessage(fit),
        paste(
          "  Tweedie p could not be estimated from the full station set for this year.",
          "  This means the model may not be viable for this year even without subsampling.",
          "  Investigate year", yr, "before proceeding:",
          "    - Check density distribution (zero fraction, max values)",
          "    - Try verbose=TRUE in the inla() call for that year",
          "    - Consider whether sdmTMB is a more robust backend",
          sep = "\n  "
        )
      ))
    }
    
    p_val <- fit$summary.hyperpar["p parameter for Tweedie", "mean"]
    p_by_year[as.character(yr)] <- p_val
    cat(sprintf("  Year %d: p = %.4f  (%.0fs)\n", yr, p_val, elapsed))
  }
  
  cat(sprintf("\n  p range across years: [%.4f, %.4f]  mean = %.4f\n",
              min(p_by_year, na.rm = TRUE),
              max(p_by_year, na.rm = TRUE),
              mean(p_by_year, na.rm = TRUE)))
  
  # Save for reproducibility / inspection
  p_df <- data.frame(year = years, p = as.numeric(p_by_year))
  write.csv(p_df, file.path(output_dir, "tweedie_p_by_year.csv"), row.names = FALSE)
  saveRDS(p_by_year, file.path(output_dir, "tweedie_p_by_year.rds"))
  cat(sprintf("  Saved to %s/tweedie_p_by_year.{{csv,rds}}\n", output_dir))
  
  return(p_by_year)  # named numeric vector: names = character years
}


# ==============================================================================
# STEP 4: FIT SPDE FOR ONE YEAR — GAUSSIAN (log space)
# ==============================================================================

fit_spde_year <- function(y, station_coords, mesh, spde, A_grid) {
  
  y_log <- log1p(y)
  A_obs <- inla.spde.make.A(mesh, loc = station_coords)
  
  stack_obs <- inla.stack(
    data    = list(y = y_log),
    A       = list(A_obs, 1),
    effects = list(spatial   = 1:spde$n.spde,
                   intercept = rep(1, length(y_log))),
    tag = "obs"
  )
  stack_pred <- inla.stack(
    data    = list(y = NA),
    A       = list(A_grid, 1),
    effects = list(spatial   = 1:spde$n.spde,
                   intercept = rep(1, nrow(A_grid))),
    tag = "pred"
  )
  stack_full <- inla.stack(stack_obs, stack_pred)
  
  result <- tryCatch({
    inla(
      y ~ -1 + intercept + f(spatial, model = spde),
      data              = inla.stack.data(stack_full),
      control.predictor = list(A = inla.stack.A(stack_full), compute = TRUE),
      control.compute   = list(config = FALSE),
      control.inla      = list(strategy = "gaussian", int.strategy = "eb"),
      verbose = FALSE
    )
  }, error = function(e) {
    warning(sprintf("INLA Gaussian failed: %s", e$message))
    return(NULL)
  })
  
  if (is.null(result)) return(rep(0, nrow(A_grid)))
  
  idx_pred <- inla.stack.index(stack_full, tag = "pred")$data
  pred_log <- result$summary.fitted.values[idx_pred, "mean"]
  pred     <- expm1(pred_log)
  pred[pred < 0]    <- 0
  pred[is.na(pred)] <- 0
  
  return(pred)
}


# ==============================================================================
# STEP 5: FIT SPDE FOR ONE YEAR — TWEEDIE (log link, raw densities)
#
# p_fixed: numeric scalar from estimate_tweedie_p_by_year().
#   When supplied, p is pinned via a near-zero-variance Normal prior on the
#   internal (logit-like) parameterisation, removing the joint estimation
#   that causes convergence failures for degenerate bootstrap subsamples.
#   When NULL, p is estimated freely (diffuse prior) — only used for the
#   full-data p-estimation fits in Step 3.
# ==============================================================================

fit_spde_year_tweedie <- function(y, station_coords, mesh, spde, A_grid,
                                  p_fixed     = NULL) {
  
  SCALE_FACTOR <- 1000
  y_scaled     <- y / SCALE_FACTOR
  obs_max      <- max(y, na.rm = TRUE)
  A_obs        <- inla.spde.make.A(mesh, loc = station_coords)
  
  stack_obs <- inla.stack(
    data    = list(y = y_scaled),
    A       = list(A_obs, 1),
    effects = list(spatial   = 1:spde$n.spde,
                   intercept = rep(1, length(y_scaled))),
    tag = "obs"
  )
  stack_pred <- inla.stack(
    data    = list(y = NA),
    A       = list(A_grid, 1),
    effects = list(spatial   = 1:spde$n.spde,
                   intercept = rep(1, nrow(A_grid))),
    tag = "pred"
  )
  stack_full <- inla.stack(stack_obs, stack_pred)
  
  # ── Build p hyperparameter prior ──────────────────────────────────────────
  # INLA's internal parameterisation for Tweedie p:
  #   theta1 = log(p - 1) - log(2 - p)   (maps p in (1,2) to the real line)
  # A near-zero variance pins p; a diffuse prior lets it be estimated freely.
  if (!is.null(p_fixed)) {
    p_internal <- log(p_fixed - 1) - log(2 - p_fixed)
    p_hyper    <- list(prior = "normal", param = c(p_internal, 0.0001))
    p_label    <- sprintf("fixed p=%.4f", p_fixed)
  } else {
    p_hyper <- list(prior = "normal", param = c(0, 10))  # diffuse
    p_label <- "free p"
  }
  
  result <- tryCatch({
    inla(
      y ~ -1 + intercept + f(spatial, model = spde),
      family = "tweedie",
      data   = inla.stack.data(stack_full),
      control.family = list(
        link  = "log",
        hyper = list(
          theta1 = p_hyper,
          theta2 = list(prior = "loggamma", param = c(1, 0.1))
        )
      ),
      control.fixed = list(
        mean = list(intercept = log(max(mean(y_scaled[y_scaled > 0], na.rm = TRUE), 1e-6))),
        prec = list(intercept = 1)
      ),
      control.predictor = list(
        A       = inla.stack.A(stack_full),
        compute = TRUE
      ),
      control.compute = list(config = FALSE),
      control.inla    = list(strategy = "gaussian", int.strategy = "eb"),
      num.threads     = "1:1",
      verbose         = FALSE
    )
  }, error = function(e) {
    warning(sprintf("INLA Tweedie failed [%s]: %s", p_label, e$message))
    return(NULL)
  })
  
  if (is.null(result)) return(rep(NA, nrow(A_grid)))
  
  cat(sprintf("    Tweedie p=%.3f (sd=%.4f), dispersion=%.4f  [%s]\n",
              result$summary.hyperpar["p parameter for Tweedie", "mean"],
              result$summary.hyperpar["p parameter for Tweedie", "sd"],
              result$summary.hyperpar["Dispersion parameter for Tweedie", "mean"],
              p_label))
  
  # ── Back-transform: exp(linear predictor) * SCALE_FACTOR ──────────────────
  idx_pred <- inla.stack.index(stack_full, tag = "pred")$data
  lp_mean  <- result$summary.linear.predictor[idx_pred, "mean"]
  
  # Clamp linear predictor to 10x observed max (guards residual blowups)
  LP_MAX  <- log(obs_max * 10 / SCALE_FACTOR)
  lp_mean <- pmin(lp_mean, LP_MAX)
  
  pred           <- exp(lp_mean) * SCALE_FACTOR
  pred[pred < 0] <- 0
  pred[is.na(pred)] <- 0
  
  cat(sprintf("    Pred range: [%.1f, %.1f]\n", min(pred), max(pred)))
  
  return(pred)
}


# ==============================================================================
# STEP 6: PARALLEL BOOTSTRAP
# p_by_year is a named numeric vector (names = character years) produced by
# estimate_tweedie_p_by_year(). It is exported to each worker and looked up
# per year so that p is fixed at the full-data estimate for every bootstrap fit.
# ==============================================================================

run_bootstrap_parallel <- function(data_sf, station_locations_sf, grid_info, spde_comps,
                                   p_by_year,
                                   n_bootstraps = N_BOOTSTRAPS, n_subsample = N_SUBSAMPLE,
                                   seed = SEED, n_cores = 4,
                                   run_label = "spawner",
                                   max_retries = 5) {
  
  years          <- sort(unique(data_sf$year))
  n_years        <- length(years)
  all_stations   <- unique(data_sf$station)
  n_stations     <- length(all_stations)
  stn_coords_all <- st_coordinates(station_locations_sf)
  stn_names_all  <- station_locations_sf$station
  
  cat(sprintf("\n  Parallel bootstrap [%s]: %d stations, %d years, %d bootstraps on %d cores\n",
              run_label, n_stations, n_years, n_bootstraps, n_cores))
  
  set.seed(seed)
  bootstrap_stations <- lapply(1:n_bootstraps, function(b) {
    sub_idx <- sample(1:n_stations, min(n_subsample, n_stations), replace = FALSE)
    all_stations[sub_idx]
  })
  
  local_env <- environment()
  
  run_one_bootstrap <- function(b) {
    library(INLA)
    library(dplyr)
    library(sf)
    library(R.utils)
    
    inla.setOption(num.threads = "1:1")
    Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
    
    log_file <- file.path(OUTPUT_DIR,
                          sprintf("worker_%s_%03d.log", run_label, b))
    
    cat(sprintf("[%s B%03d] Started at %s\n", run_label, b, Sys.time()),
        file = log_file, append = FALSE)
    
    tryCatch({
      
      sub_stations <- bootstrap_stations[[b]]
      result_array <- array(NA_real_, dim = c(n_years, PAD_NY, PAD_NX))
      
      for (y_idx in 1:n_years) {
        yr      <- years[y_idx]
        yr_data <- data_sf %>% filter(year == yr, station %in% sub_stations)
        
        if (nrow(yr_data) == 0) {
          cat(sprintf("[%s B%03d] Year %d SKIPPED (no data)\n", run_label, b, yr),
              file = log_file, append = TRUE)
          next
        }
        
        avail_stations <- yr_data$station
        avail_coords   <- stn_coords_all[
          match(avail_stations, stn_names_all), , drop = FALSE
        ]
        
        p_fixed_yr <- p_by_year[as.character(yr)]
        
        # ── Retry loop ──────────────────────────────────────────────────────
        pred_valid <- NULL
        attempt    <- 0
        
        while (attempt < max_retries) {
          attempt <- attempt + 1
          
          t0 <- proc.time()["elapsed"]
          
          pred_valid <- tryCatch({
            fit_spde_year_tweedie(
              y              = yr_data$avg_dens,
              station_coords = avail_coords,
              mesh           = spde_comps$mesh,
              spde           = spde_comps$spde,
              A_grid         = spde_comps$A_grid,
              p_fixed        = p_fixed_yr
            )
          }, error = function(e) {
            cat(sprintf("[%s B%03d] Year %d attempt %d FAILED: %s\n",
                        run_label, b, yr, attempt, e$message),
                file = log_file, append = TRUE)
            NULL
          })
          
          elapsed <- proc.time()["elapsed"] - t0
          
          if (!is.null(pred_valid) && !all(is.na(pred_valid)) && any(pred_valid > 0, na.rm = TRUE)) {
            cat(sprintf("[%s B%03d] Year %d attempt %d SUCCESS in %.0fs [pred: %.0f-%.0f]\n",
                        run_label, b, yr, attempt, elapsed,
                        min(pred_valid, na.rm = TRUE), max(pred_valid, na.rm = TRUE)),
                file = log_file, append = TRUE)
            break
          }
          
          cat(sprintf("[%s B%03d] Year %d attempt %d no valid result — retrying...\n",
                      run_label, b, yr, attempt),
              file = log_file, append = TRUE)
          pred_valid <- NULL
        }
        # ── End retry loop ───────────────────────────────────────────────────
        
        if (is.null(pred_valid)) {
          cat(sprintf("[%s B%03d] Year %d EXHAUSTED %d retries — leaving slice as NA\n",
                      run_label, b, yr, max_retries),
              file = log_file, append = TRUE)
          next
        }
        
        # ── Blowup tracker ──
        obs_max_yr  <- max(data_sf$avg_dens[data_sf$year == yr], na.rm = TRUE)
        max_pred_yr <- max(pred_valid, na.rm = TRUE)
        
        if (max_pred_yr > 10 * obs_max_yr) {
          cat(sprintf("[%s B%03d] Year %d BLOWUP: max_pred=%.2e vs obs_max=%.2e (%.1fx)\n",
                      run_label, b, yr,
                      max_pred_yr, obs_max_yr, max_pred_yr / obs_max_yr),
              file = log_file, append = TRUE)
        }
        
        cat(sprintf("[%s B%03d] Year %d done in %.0fs [pred: %.0f-%.0f]  p=%.4f\n",
                    run_label, b, yr, elapsed,
                    min(pred_valid, na.rm = TRUE), max(pred_valid, na.rm = TRUE), p_fixed_yr),
            file = log_file, append = TRUE)
        
        result_array[y_idx, , ] <- fill_matrix(pred_valid, grid_info)
      }
      
      # ── Checkpoint: save each bootstrap independently ──
      saveRDS(result_array,
              file.path(OUTPUT_DIR, sprintf("checkpoint_%s_b%03d.rds", run_label, b)))
      
      cat(sprintf("[%s B%03d] COMPLETE at %s\n", run_label, b, Sys.time()),
          file = log_file, append = TRUE)
      
      return(result_array)
      
    }, error = function(e) {
      cat(sprintf("[%s B%03d] WORKER CRASH: %s\n", run_label, b, e$message),
          file = log_file, append = TRUE)
      return(array(NA_real_, dim = c(n_years, PAD_NY, PAD_NX)))
    })
  }
  
  cat("  Launching workers...\n")
  start_time <- Sys.time()
  
  cl <- makeCluster(n_cores)
  
  clusterExport(cl, varlist = c(
    "bootstrap_stations",
    "data_sf",
    "stn_coords_all",
    "stn_names_all",
    "all_stations",
    "years",
    "n_years",
    "grid_info",
    "spde_comps",
    "p_by_year",
    "fit_spde_year_tweedie",
    "fill_matrix",
    "PAD_NY",
    "PAD_NX",
    "OUTPUT_DIR",
    "run_label",
    "max_retries"
  ), envir = local_env)
  
  clusterEvalQ(cl, library(R.utils))
  
  results_list <- parLapply(cl, 1:n_bootstraps, run_one_bootstrap)
  stopCluster(cl)
  
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  cat(sprintf("  Done! %d bootstraps in %.1f min\n", n_bootstraps, elapsed))
  
  # ── Reassemble from checkpoints ──
  output <- array(NA_real_, dim = c(n_bootstraps, n_years, PAD_NY, PAD_NX))
  for (b in 1:n_bootstraps) {
    ckpt <- file.path(OUTPUT_DIR, sprintf("checkpoint_%s_b%03d.rds", run_label, b))
    if (file.exists(ckpt)) {
      output[b, , , ] <- readRDS(ckpt)
    } else if (!is.null(results_list[[b]])) {
      output[b, , , ] <- results_list[[b]]
    } else {
      cat(sprintf("  WARNING: %s bootstrap %d missing!\n", run_label, b))
    }
  }
  
  # ── Summarise blowups across all worker logs ──
  cat(sprintf("\n  Scanning %s logs for blowups...\n", run_label))
  log_files    <- list.files(OUTPUT_DIR,
                             pattern = sprintf("worker_%s_.*\\.log", run_label),
                             full.names = TRUE)
  blowup_lines <- unlist(lapply(log_files, function(f) {
    lines <- readLines(f, warn = FALSE)
    lines[grep("BLOWUP", lines)]
  }))
  
  if (length(blowup_lines) == 0) {
    cat(sprintf("  No blowups detected across %d %s bootstraps.\n",
                n_bootstraps, run_label))
  } else {
    cat(sprintf("  %d blowup(s) detected in %s run:\n",
                length(blowup_lines), run_label))
    cat(paste(" ", blowup_lines, collapse = "\n"), "\n")
  }
  
  return(list(data = output, years = years))
}


# ==============================================================================
# STEP 7: RERUN MISSING BOOTSTRAPS
# ==============================================================================

rerun_missing <- function(missing_bs, data_sf, station_locations_sf,
                          grid_info, spde_comps, p_by_year,
                          seed = SEED, n_subsample = N_SUBSAMPLE, n_cores = 3,
                          run_label = "spawner") {
  
  all_stations <- unique(data_sf$station)
  n_stations   <- length(all_stations)
  
  # Regenerate full original draws so seeds match
  set.seed(seed)
  all_bootstrap_stations <- lapply(1:100, function(b) {
    sub_idx <- sample(1:n_stations, min(n_subsample, n_stations), replace = FALSE)
    all_stations[sub_idx]
  })
  
  bootstrap_stations <- all_bootstrap_stations[missing_bs]
  years   <- sort(unique(data_sf$year))
  n_years <- length(years)
  
  stn_coords_all <- st_coordinates(station_locations_sf)
  stn_names_all  <- station_locations_sf$station
  
  local_env <- environment()
  
  run_one <- function(i) {
    library(INLA); library(dplyr); library(sf); library(R.utils)
    inla.setOption(num.threads = "1:1")
    Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
    
    b        <- missing_bs[i]
    log_file <- file.path(OUTPUT_DIR, sprintf("worker_%s_%03d.log", run_label, b))
    cat(sprintf("[%s B%03d] RERUN started at %s\n", run_label, b, Sys.time()),
        file = log_file, append = TRUE)
    
    tryCatch({
      sub_stations <- bootstrap_stations[[i]]
      result_array <- array(0, dim = c(n_years, PAD_NY, PAD_NX))
      
      for (y_idx in 1:n_years) {
        yr      <- years[y_idx]
        yr_data <- data_sf %>% filter(year == yr, station %in% sub_stations)
        if (nrow(yr_data) == 0) next
        
        avail_coords <- stn_coords_all[
          match(yr_data$station, stn_names_all), , drop = FALSE
        ]
        
        p_fixed_yr <- p_by_year[as.character(yr)]
        
        t0 <- proc.time()["elapsed"]
        
        pred_valid <- tryCatch({
          fit_spde_year_tweedie(yr_data$avg_dens, avail_coords,
                                spde_comps$mesh, spde_comps$spde,
                                spde_comps$A_grid,
                                p_fixed     = p_fixed_yr)
        }, error = function(e) {
          cat(sprintf("[%s B%03d] Year %d FAILED: %s\n", run_label, b, yr, e$message),
              file = log_file, append = TRUE)
          rep(NA, nrow(spde_comps$A_grid))
        })
        
        elapsed <- proc.time()["elapsed"] - t0
        
        if (all(is.na(pred_valid))) {
          cat(sprintf("[%s B%03d] Year %d TIMEOUT — filling zeros\n", run_label, b, yr),
              file = log_file, append = TRUE)
          pred_valid <- rep(0, nrow(spde_comps$A_grid))
        }
        
        obs_max_yr  <- max(data_sf$avg_dens[data_sf$year == yr], na.rm = TRUE)
        max_pred_yr <- max(pred_valid, na.rm = TRUE)
        if (max_pred_yr > 10 * obs_max_yr) {
          cat(sprintf("[%s B%03d] Year %d BLOWUP: max_pred=%.2e vs obs_max=%.2e (%.1fx)\n",
                      run_label, b, yr, max_pred_yr, obs_max_yr, max_pred_yr / obs_max_yr),
              file = log_file, append = TRUE)
        }
        
        cat(sprintf("[%s B%03d] Year %d done in %.0fs [pred: %.0f-%.0f]  p=%.4f\n",
                    run_label, b, yr, elapsed,
                    min(pred_valid), max(pred_valid), p_fixed_yr),
            file = log_file, append = TRUE)
        
        result_array[y_idx, , ] <- fill_matrix(pred_valid, grid_info)
      }
      
      saveRDS(result_array,
              file.path(OUTPUT_DIR, sprintf("checkpoint_%s_b%03d.rds", run_label, b)))
      cat(sprintf("[%s B%03d] RERUN COMPLETE at %s\n", run_label, b, Sys.time()),
          file = log_file, append = TRUE)
      return(result_array)
      
    }, error = function(e) {
      cat(sprintf("[%s B%03d] RERUN CRASH: %s\n", run_label, b, e$message),
          file = log_file, append = TRUE)
      return(array(0, dim = c(n_years, PAD_NY, PAD_NX)))
    })
  }
  
  cat(sprintf("Rerunning %d missing bootstraps on %d cores...\n",
              length(missing_bs), min(n_cores, length(missing_bs))))
  
  cl <- makeCluster(min(n_cores, length(missing_bs)))
  
  clusterExport(cl, varlist = c(
    "missing_bs", "bootstrap_stations",
    "data_sf", "stn_coords_all", "stn_names_all",
    "years", "n_years", "grid_info", "spde_comps",
    "p_by_year",                  # ← new
    "fit_spde_year_tweedie", "fill_matrix",
    "PAD_NY", "PAD_NX", "OUTPUT_DIR", "run_label"
  ), envir = local_env)
  
  clusterEvalQ(cl, library(R.utils))
  
  results <- parLapply(cl, seq_along(missing_bs), run_one)
  stopCluster(cl)
  
  cat("Rerun complete.\n")
  return(results)
}


# ==============================================================================
# STEP 8: SAVE OUTPUTS
# ==============================================================================

save_outputs <- function(spawner_result, recruit_result, grid_info,
                         output_dir = OUTPUT_DIR) {
  
  mask <- fill_matrix(rep(1, grid_info$n_valid), grid_info)
  
  saveRDS(spawner_result$data, file.path(output_dir, "gridded_spawners.rds"))
  saveRDS(recruit_result$data, file.path(output_dir, "gridded_recruits.rds"))
  saveRDS(mask,                file.path(output_dir, "spatial_mask.rds"))
  write.csv(mask, file.path(output_dir, "spatial_mask.csv"), row.names = FALSE)
  
  if (requireNamespace("reticulate", quietly = TRUE)) {
    np <- reticulate::import("numpy")
    np$save(file.path(output_dir, "gridded_spawners.npy"), spawner_result$data)
    np$save(file.path(output_dir, "gridded_recruits.npy"), recruit_result$data)
    np$save(file.path(output_dir, "spatial_mask.npy"),     mask)
    cat("Saved .npy files\n")
  }
  
  metadata <- list(
    cellsize_km     = grid_info$cellsize,
    nx_full         = grid_info$nx_full,
    ny_full         = grid_info$ny_full,
    pad_nx          = PAD_NX,
    pad_ny          = PAD_NY,
    n_valid_cells   = grid_info$n_valid,
    n_bootstraps    = dim(spawner_result$data)[1],
    spawner_years   = spawner_result$years,
    recruit_years   = recruit_result$years,
    n_spawner_years = length(spawner_result$years),
    n_recruit_years = length(recruit_result$years),
    crs             = "+proj=utm +zone=2 +datum=WGS84 +units=km",
    bbox_xmin       = as.numeric(grid_info$bbox["xmin"]),
    bbox_xmax       = as.numeric(grid_info$bbox["xmax"]),
    bbox_ymin       = as.numeric(grid_info$bbox["ymin"]),
    bbox_ymax       = as.numeric(grid_info$bbox["ymax"])
  )
  
  write_json(metadata, file.path(output_dir, "grid_metadata.json"),
             pretty = TRUE, auto_unbox = TRUE)
  
  cat(sprintf("All outputs saved to %s/\n", output_dir))
  cat(sprintf("  Mask: %d valid cells in %dx%d grid\n", sum(mask), PAD_NY, PAD_NX))
}


# ==============================================================================
# STEP 9: VISUALIZATION
# ==============================================================================

plot_gridded_field <- function(pred_valid, grid_info, survey_domain, sf_maps,
                               title = "SPDE Gridded Field") {
  plot_sf <- st_sf(geometry = grid_info$valid_grid, val = pred_valid)
  
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
  
  n_boot <- dim(spawner_result$data)[1]
  n_year <- dim(spawner_result$data)[2]
  if (is.null(yr_idx)) yr_idx <- ceiling(n_year / 2)
  target_year <- spawner_result$years[yr_idx]
  
  extract_valid <- function(mat, grid_info) {
    vals <- numeric(grid_info$n_valid)
    for (i in 1:grid_info$n_valid) {
      vals[i] <- mat[grid_info$grid_row[i], grid_info$grid_col[i]]
    }
    return(vals)
  }
  
  mean_s <- apply(spawner_result$data[, yr_idx, , , drop = FALSE], c(3, 4), mean)
  mean_r <- apply(recruit_result$data[, yr_idx, , , drop = FALSE], c(3, 4), mean)
  
  set.seed(42)
  rand_b <- sample(1:n_boot, 1)
  rand_s <- spawner_result$data[rand_b, yr_idx, , ]
  rand_r <- recruit_result$data[rand_b, yr_idx, , ]
  
  library(patchwork)
  
  p1 <- plot_gridded_field(extract_valid(mean_s, grid_info), grid_info,
                           survey_domain, sf_maps,
                           sprintf("Spawner Mean \u2014 Year %d", target_year))
  p2 <- plot_gridded_field(extract_valid(rand_s, grid_info), grid_info,
                           survey_domain, sf_maps,
                           sprintf("Spawner Boot %d \u2014 Year %d", rand_b, target_year))
  p3 <- plot_gridded_field(extract_valid(mean_r, grid_info), grid_info,
                           survey_domain, sf_maps,
                           sprintf("Recruit Mean \u2014 Year %d", target_year))
  p4 <- plot_gridded_field(extract_valid(rand_r, grid_info), grid_info,
                           survey_domain, sf_maps,
                           sprintf("Recruit Boot %d \u2014 Year %d", rand_b, target_year))
  
  combined <- (p1 + p2) / (p3 + p4)
  print(combined)
  ggsave(file.path(OUTPUT_DIR, "bootstrap_comparison.png"), combined,
         width = 14, height = 12, dpi = 150)
}


plot_mask_check <- function(grid_info, survey_domain, station_locations_sf, sf_maps) {
  
  mask_sf <- st_sf(geometry = grid_info$valid_grid, valid = 1)
  
  p <- ggplot() +
    geom_sf(data = mask_sf, fill = "steelblue", alpha = 0.3,
            color = "steelblue", lwd = 0.2) +
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
                         n_subsample  = N_SUBSAMPLE,
                         cellsize     = CELLSIZE,
                         n_cores      = 4) {
  
  cat("=== SPDE Bootstrap Gridding Pipeline ===\n\n")
  
  if (!"station" %in% names(station_locations_sf)) {
    station_locations_sf <- station_locations_sf %>% rename(station = GIS_STATION)
  }
  
  cat(sprintf("Spawner: %d-%d (%d years)\n",
              min(spawner_sf$year), max(spawner_sf$year),
              length(unique(spawner_sf$year))))
  cat(sprintf("Recruit: %d-%d (%d years)\n",
              min(recruit_sf$year), max(recruit_sf$year),
              length(unique(recruit_sf$year))))
  
  cat("\n[1/6] Building grid...\n")
  grid_info <- build_grid(survey_domain, station_locations_sf, cellsize)
  
  cat("\n[2/6] Building SPDE mesh...\n")
  stn_coords <- st_coordinates(station_locations_sf)
  spde_comps <- build_spde(stn_coords, grid_info)
  
  # ── New: estimate p per year before bootstrapping ──────────────────────────
  cat("\n[3/6] Estimating Tweedie p by year (spawners)...\n")
  spawner_p_by_year <- estimate_tweedie_p_by_year(
    spawner_sf, station_locations_sf, spde_comps, output_dir = OUTPUT_DIR
  )
  
  cat("\n[3b/6] Estimating Tweedie p by year (recruits)...\n")
  recruit_p_by_year <- estimate_tweedie_p_by_year(
    recruit_sf, station_locations_sf, spde_comps, output_dir = OUTPUT_DIR
  )
  # Save recruit p separately so they don't overwrite spawner's
  saveRDS(recruit_p_by_year,
          file.path(OUTPUT_DIR, "tweedie_p_by_year_recruit.rds"))
  
  cat("\n[Sanity] Fitting one test year...\n")
  test_yr     <- sort(unique(spawner_sf$year))[15]
  test_data   <- spawner_sf %>% filter(year == test_yr)
  test_coords <- stn_coords[match(test_data$station,
                                  station_locations_sf$station), , drop = FALSE]
  test_pred   <- fit_spde_year_tweedie(
    test_data$avg_dens, test_coords,
    spde_comps$mesh, spde_comps$spde, spde_comps$A_grid,
    p_fixed = spawner_p_by_year[as.character(test_yr)]
  )
  cat(sprintf("  Year %d: range [%.0f, %.0f], nonzero: %d/%d\n",
              test_yr, min(test_pred), max(test_pred),
              sum(test_pred > 0), length(test_pred)))
  if (!is.null(sf_maps)) {
    print(plot_gridded_field(test_pred, grid_info, survey_domain, sf_maps,
                             sprintf("Sanity Check \u2014 Year %d", test_yr)))
  }
  
  cat("\n[4/6] Gridding spawners...\n")
  spawner_result <- run_bootstrap_parallel(
    spawner_sf, station_locations_sf, grid_info, spde_comps,
    p_by_year    = spawner_p_by_year,
    n_bootstraps = n_bootstraps, n_subsample = n_subsample,
    seed = SEED, n_cores = n_cores, run_label = "spawner"
  )
  
  cat("\n[5/6] Gridding recruits...\n")
  recruit_result <- run_bootstrap_parallel(
    recruit_sf, station_locations_sf, grid_info, spde_comps,
    p_by_year    = recruit_p_by_year,
    n_bootstraps = n_bootstraps, n_subsample = n_subsample,
    seed = SEED + 1000, n_cores = n_cores, run_label = "recruit"
  )
  
  cat("\n[6/6] Saving...\n")
  save_outputs(spawner_result, recruit_result, grid_info)
  
  if (!is.null(sf_maps)) {
    plot_mask_check(grid_info, survey_domain, station_locations_sf, sf_maps)
    plot_year_comparison(spawner_result, recruit_result, grid_info,
                         survey_domain, sf_maps)
  }
  
  return(list(
    spawner           = spawner_result,
    recruit           = recruit_result,
    grid_info         = grid_info,
    spde_comps        = spde_comps,
    spawner_p_by_year = spawner_p_by_year,
    recruit_p_by_year = recruit_p_by_year
  ))
}


# ==============================================================================
# RUN
# ==============================================================================
library(ggplot2)
station_locations_sf_named <- station_locations_sf %>%
  rename(station = GIS_STATION)

test <- run_gridding(
  spawner_sf, recruit_sf, station_locations_sf_named,
  survey_domain, sf_maps,
  n_bootstraps = 100,
  n_subsample  = 250,
  n_cores      = 4
)

ckpts     <- list.files(OUTPUT_DIR, pattern = "checkpoint_spawner_.*\\.rds")
completed <- sort(as.integer(gsub(".*b(\\d+)\\.rds", "\\1", ckpts)))
missing   <- sort(setdiff(1:100, completed))
cat(sprintf("Completed: %d / 100\n", length(completed)))
cat(sprintf("Missing: %s\n", paste(missing, collapse = ", ")))

# Load the already-estimated p values (no need to re-estimate)
spawner_p_by_year <- readRDS(file.path(OUTPUT_DIR, "tweedie_p_by_year.rds"))

grid_info  <- build_grid(survey_domain, station_locations_sf_named, CELLSIZE)
spde_comps <- build_spde(st_coordinates(station_locations_sf_named), grid_info)

rerun_missing(
  missing_bs           = missing,
  data_sf              = spawner_sf,
  station_locations_sf = station_locations_sf_named,
  grid_info            = grid_info,
  spde_comps           = spde_comps,
  p_by_year            = spawner_p_by_year,
  n_cores              = 1,
  run_label            = "spawner"
  
)

# ── Insert 2020 as zeros (no survey that year) ──
all_years     <- 1988:2023
n_years_new   <- length(all_years)          # 36
n_years_orig  <- dim(test$spawner$data)[2]  # 35
n_boot        <- dim(test$spawner$data)[1]
year_2020_idx <- which(all_years == 2020)   # 33

spawner_grids_new <- array(0, dim = c(n_boot, n_years_new, 50, 50))
recruit_grids_new <- array(0, dim = c(n_boot, n_years_new, 50, 50))

spawner_grids_new[, 1:(year_2020_idx - 1), , ] <- test$spawner$data[, 1:(year_2020_idx - 1), , ]
recruit_grids_new[, 1:(year_2020_idx - 1), , ] <- test$recruit$data[, 1:(year_2020_idx - 1), , ]

# Index 33 stays zero (2020 — no survey)

spawner_grids_new[, (year_2020_idx + 1):n_years_new, , ] <- test$spawner$data[, year_2020_idx:n_years_orig, , ]
recruit_grids_new[, (year_2020_idx + 1):n_years_new, , ] <- test$recruit$data[, year_2020_idx:n_years_orig, , ]

year_mask                <- rep(1L, n_years_new)
year_mask[year_2020_idx] <- 0L

mask <- fill_matrix(rep(1, test$grid_info$n_valid), test$grid_info)

saveRDS(spawner_grids_new, file.path(OUTPUT_DIR, "gridded_spawners.rds"))
saveRDS(recruit_grids_new, file.path(OUTPUT_DIR, "gridded_recruits.rds"))
saveRDS(mask,              file.path(OUTPUT_DIR, "spatial_mask.rds"))
saveRDS(year_mask,         file.path(OUTPUT_DIR, "year_mask.rds"))
saveRDS(all_years,         file.path(OUTPUT_DIR, "years.rds"))

if (requireNamespace("reticulate", quietly = TRUE)) {
  np <- reticulate::import("numpy")
  np$save(file.path(OUTPUT_DIR, "gridded_spawners.npy"), spawner_grids_new)
  np$save(file.path(OUTPUT_DIR, "gridded_recruits.npy"), recruit_grids_new)
  np$save(file.path(OUTPUT_DIR, "spatial_mask.npy"),     mask)
  np$save(file.path(OUTPUT_DIR, "year_mask.npy"),        year_mask)
  np$save(file.path(OUTPUT_DIR, "years.npy"),            as.integer(all_years))
  cat("Saved .npy files\n")
}














# ==============================================================================
# GAUSSIAN PIPELINE — parallel bootstrap + rerun (no p-estimation needed)
# Drop-in companion to the Tweedie pipeline; shares grid_info and spde_comps.
# ==============================================================================

run_bootstrap_parallel_gaussian <- function(data_sf, station_locations_sf,
                                            grid_info, spde_comps,
                                            n_bootstraps = N_BOOTSTRAPS,
                                            n_subsample  = N_SUBSAMPLE,
                                            seed         = SEED,
                                            n_cores      = 4,
                                            run_label    = "spawner",
                                            max_retries  = 5) {
  
  years          <- sort(unique(data_sf$year))
  n_years        <- length(years)
  all_stations   <- unique(data_sf$station)
  n_stations     <- length(all_stations)
  stn_coords_all <- st_coordinates(station_locations_sf)
  stn_names_all  <- station_locations_sf$station
  
  cat(sprintf(
    "\n  Parallel Gaussian bootstrap [%s]: %d stations, %d years, %d bootstraps on %d cores\n",
    run_label, n_stations, n_years, n_bootstraps, n_cores
  ))
  
  set.seed(seed)
  bootstrap_stations <- lapply(1:n_bootstraps, function(b) {
    sub_idx <- sample(1:n_stations, min(n_subsample, n_stations), replace = FALSE)
    all_stations[sub_idx]
  })
  
  local_env <- environment()
  
  run_one_bootstrap <- function(b) {
    library(INLA)
    library(dplyr)
    library(sf)
    
    inla.setOption(num.threads = "1:1")
    Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
    
    log_file <- file.path(OUTPUT_DIR,
                          sprintf("worker_%s_gauss_%03d.log", run_label, b))
    cat(sprintf("[%s-G B%03d] Started at %s\n", run_label, b, Sys.time()),
        file = log_file, append = FALSE)
    
    tryCatch({
      
      sub_stations <- bootstrap_stations[[b]]
      result_array <- array(NA_real_, dim = c(n_years, PAD_NY, PAD_NX))
      
      for (y_idx in 1:n_years) {
        yr      <- years[y_idx]
        yr_data <- data_sf %>% filter(year == yr, station %in% sub_stations)
        
        if (nrow(yr_data) == 0) {
          cat(sprintf("[%s-G B%03d] Year %d SKIPPED (no data)\n",
                      run_label, b, yr),
              file = log_file, append = TRUE)
          next
        }
        
        avail_coords <- stn_coords_all[
          match(yr_data$station, stn_names_all), , drop = FALSE
        ]
        
        pred_valid <- NULL
        attempt    <- 0
        
        while (attempt < max_retries) {
          attempt <- attempt + 1
          t0      <- proc.time()["elapsed"]
          
          pred_valid <- tryCatch({
            fit_spde_year(
              y              = yr_data$avg_dens,
              station_coords = avail_coords,
              mesh           = spde_comps$mesh,
              spde           = spde_comps$spde,
              A_grid         = spde_comps$A_grid
            )
          }, error = function(e) {
            cat(sprintf("[%s-G B%03d] Year %d attempt %d FAILED: %s\n",
                        run_label, b, yr, attempt, e$message),
                file = log_file, append = TRUE)
            NULL
          })
          
          elapsed <- proc.time()["elapsed"] - t0
          
          if (!is.null(pred_valid) &&
              !all(is.na(pred_valid)) &&
              any(pred_valid > 0, na.rm = TRUE)) {
            cat(sprintf(
              "[%s-G B%03d] Year %d attempt %d SUCCESS in %.0fs [pred: %.0f-%.0f]\n",
              run_label, b, yr, attempt, elapsed,
              min(pred_valid, na.rm = TRUE), max(pred_valid, na.rm = TRUE)
            ), file = log_file, append = TRUE)
            break
          }
          
          cat(sprintf("[%s-G B%03d] Year %d attempt %d no valid result — retrying...\n",
                      run_label, b, yr, attempt),
              file = log_file, append = TRUE)
          pred_valid <- NULL
        }
        
        if (is.null(pred_valid)) {
          cat(sprintf("[%s-G B%03d] Year %d EXHAUSTED %d retries — leaving slice NA\n",
                      run_label, b, yr, max_retries),
              file = log_file, append = TRUE)
          next
        }
        
        # Blowup check (log-space model, so 10x observed is still a useful guard)
        obs_max_yr  <- max(data_sf$avg_dens[data_sf$year == yr], na.rm = TRUE)
        max_pred_yr <- max(pred_valid, na.rm = TRUE)
        if (max_pred_yr > 10 * obs_max_yr) {
          cat(sprintf(
            "[%s-G B%03d] Year %d BLOWUP: max_pred=%.2e vs obs_max=%.2e (%.1fx)\n",
            run_label, b, yr, max_pred_yr, obs_max_yr, max_pred_yr / obs_max_yr
          ), file = log_file, append = TRUE)
        }
        
        cat(sprintf("[%s-G B%03d] Year %d done in %.0fs [pred: %.0f-%.0f]\n",
                    run_label, b, yr, elapsed,
                    min(pred_valid, na.rm = TRUE), max(pred_valid, na.rm = TRUE)),
            file = log_file, append = TRUE)
        
        result_array[y_idx, , ] <- fill_matrix(pred_valid, grid_info)
      }
      
      saveRDS(result_array,
              file.path(OUTPUT_DIR,
                        sprintf("checkpoint_%s_gauss_b%03d.rds", run_label, b)))
      cat(sprintf("[%s-G B%03d] COMPLETE at %s\n", run_label, b, Sys.time()),
          file = log_file, append = TRUE)
      
      return(result_array)
      
    }, error = function(e) {
      cat(sprintf("[%s-G B%03d] WORKER CRASH: %s\n", run_label, b, e$message),
          file = log_file, append = TRUE)
      return(array(NA_real_, dim = c(n_years, PAD_NY, PAD_NX)))
    })
  }
  
  cat("  Launching Gaussian workers...\n")
  start_time <- Sys.time()
  
  cl <- makeCluster(n_cores)
  clusterExport(cl, varlist = c(
    "bootstrap_stations",
    "data_sf",
    "stn_coords_all",
    "stn_names_all",
    "all_stations",
    "years",
    "n_years",
    "grid_info",
    "spde_comps",
    "fit_spde_year",      # Gaussian fitter — no p_by_year needed
    "fill_matrix",
    "PAD_NY",
    "PAD_NX",
    "OUTPUT_DIR",
    "run_label",
    "max_retries"
  ), envir = local_env)
  
  results_list <- parLapply(cl, 1:n_bootstraps, run_one_bootstrap)
  stopCluster(cl)
  
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  cat(sprintf("  Done! %d Gaussian bootstraps in %.1f min\n", n_bootstraps, elapsed))
  
  # Reassemble from checkpoints
  output <- array(NA_real_, dim = c(n_bootstraps, n_years, PAD_NY, PAD_NX))
  for (b in 1:n_bootstraps) {
    ckpt <- file.path(OUTPUT_DIR,
                      sprintf("checkpoint_%s_gauss_b%03d.rds", run_label, b))
    if (file.exists(ckpt)) {
      output[b, , , ] <- readRDS(ckpt)
    } else if (!is.null(results_list[[b]])) {
      output[b, , , ] <- results_list[[b]]
    } else {
      cat(sprintf("  WARNING: Gaussian %s bootstrap %d missing!\n", run_label, b))
    }
  }
  
  # Scan logs for blowups
  cat(sprintf("\n  Scanning %s Gaussian logs for blowups...\n", run_label))
  log_files    <- list.files(OUTPUT_DIR,
                             pattern = sprintf("worker_%s_gauss_.*\\.log", run_label),
                             full.names = TRUE)
  blowup_lines <- unlist(lapply(log_files, function(f) {
    lines <- readLines(f, warn = FALSE)
    lines[grep("BLOWUP", lines)]
  }))
  
  if (length(blowup_lines) == 0) {
    cat(sprintf("  No blowups in %d Gaussian %s bootstraps.\n",
                n_bootstraps, run_label))
  } else {
    cat(sprintf("  %d blowup(s) in Gaussian %s run:\n",
                length(blowup_lines), run_label))
    cat(paste(" ", blowup_lines, collapse = "\n"), "\n")
  }
  
  return(list(data = output, years = years))
}


rerun_missing_gaussian <- function(missing_bs, data_sf, station_locations_sf,
                                   grid_info, spde_comps,
                                   seed      = SEED,
                                   n_subsample = N_SUBSAMPLE,
                                   n_cores   = 3,
                                   run_label = "spawner") {
  
  all_stations <- unique(data_sf$station)
  n_stations   <- length(all_stations)
  
  # Regenerate full original draws so indices match
  set.seed(seed)
  all_bootstrap_stations <- lapply(1:100, function(b) {
    sub_idx <- sample(1:n_stations, min(n_subsample, n_stations), replace = FALSE)
    all_stations[sub_idx]
  })
  bootstrap_stations <- all_bootstrap_stations[missing_bs]
  
  years          <- sort(unique(data_sf$year))
  n_years        <- length(years)
  stn_coords_all <- st_coordinates(station_locations_sf)
  stn_names_all  <- station_locations_sf$station
  
  local_env <- environment()
  
  run_one <- function(i) {
    library(INLA); library(dplyr); library(sf)
    inla.setOption(num.threads = "1:1")
    Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
    
    b        <- missing_bs[i]
    log_file <- file.path(OUTPUT_DIR,
                          sprintf("worker_%s_gauss_%03d.log", run_label, b))
    cat(sprintf("[%s-G B%03d] RERUN started at %s\n", run_label, b, Sys.time()),
        file = log_file, append = TRUE)
    
    tryCatch({
      sub_stations <- bootstrap_stations[[i]]
      result_array <- array(0, dim = c(n_years, PAD_NY, PAD_NX))
      
      for (y_idx in 1:n_years) {
        yr      <- years[y_idx]
        yr_data <- data_sf %>% filter(year == yr, station %in% sub_stations)
        if (nrow(yr_data) == 0) next
        
        avail_coords <- stn_coords_all[
          match(yr_data$station, stn_names_all), , drop = FALSE
        ]
        
        t0 <- proc.time()["elapsed"]
        
        pred_valid <- tryCatch({
          fit_spde_year(
            y              = yr_data$avg_dens,
            station_coords = avail_coords,
            mesh           = spde_comps$mesh,
            spde           = spde_comps$spde,
            A_grid         = spde_comps$A_grid
          )
        }, error = function(e) {
          cat(sprintf("[%s-G B%03d] Year %d FAILED: %s\n",
                      run_label, b, yr, e$message),
              file = log_file, append = TRUE)
          rep(NA, nrow(spde_comps$A_grid))
        })
        
        elapsed <- proc.time()["elapsed"] - t0
        
        if (all(is.na(pred_valid))) {
          cat(sprintf("[%s-G B%03d] Year %d FAILED — filling zeros\n",
                      run_label, b, yr),
              file = log_file, append = TRUE)
          pred_valid <- rep(0, nrow(spde_comps$A_grid))
        }
        
        obs_max_yr  <- max(data_sf$avg_dens[data_sf$year == yr], na.rm = TRUE)
        max_pred_yr <- max(pred_valid, na.rm = TRUE)
        if (max_pred_yr > 10 * obs_max_yr) {
          cat(sprintf(
            "[%s-G B%03d] Year %d BLOWUP: max_pred=%.2e vs obs_max=%.2e (%.1fx)\n",
            run_label, b, yr, max_pred_yr, obs_max_yr, max_pred_yr / obs_max_yr
          ), file = log_file, append = TRUE)
        }
        
        cat(sprintf("[%s-G B%03d] Year %d done in %.0fs [pred: %.0f-%.0f]\n",
                    run_label, b, yr, elapsed,
                    min(pred_valid), max(pred_valid)),
            file = log_file, append = TRUE)
        
        result_array[y_idx, , ] <- fill_matrix(pred_valid, grid_info)
      }
      
      saveRDS(result_array,
              file.path(OUTPUT_DIR,
                        sprintf("checkpoint_%s_gauss_b%03d.rds", run_label, b)))
      cat(sprintf("[%s-G B%03d] RERUN COMPLETE at %s\n", run_label, b, Sys.time()),
          file = log_file, append = TRUE)
      return(result_array)
      
    }, error = function(e) {
      cat(sprintf("[%s-G B%03d] RERUN CRASH: %s\n", run_label, b, e$message),
          file = log_file, append = TRUE)
      return(array(0, dim = c(n_years, PAD_NY, PAD_NX)))
    })
  }
  
  cat(sprintf("Rerunning %d missing Gaussian bootstraps on %d cores...\n",
              length(missing_bs), min(n_cores, length(missing_bs))))
  
  cl <- makeCluster(min(n_cores, length(missing_bs)))
  clusterExport(cl, varlist = c(
    "missing_bs", "bootstrap_stations",
    "data_sf", "stn_coords_all", "stn_names_all",
    "years", "n_years", "grid_info", "spde_comps",
    "fit_spde_year", "fill_matrix",
    "PAD_NY", "PAD_NX", "OUTPUT_DIR", "run_label"
  ), envir = local_env)
  
  results <- parLapply(cl, seq_along(missing_bs), run_one)
  stopCluster(cl)
  
  cat("Gaussian rerun complete.\n")
  return(results)
}


# Gaussian spawners
spawner_result_gauss <- run_bootstrap_parallel_gaussian(
  spawner_sf, station_locations_sf_named, grid_info, spde_comps,
  n_bootstraps = 100, n_subsample = 300,
  seed = SEED, n_cores = 4, run_label = "spawner"
)

# Gaussian recruits
recruit_result_gauss <- run_bootstrap_parallel_gaussian(
  recruit_sf, station_locations_sf_named, grid_info, spde_comps,
  n_bootstraps = 100, n_subsample = 300,
  seed = SEED, n_cores = 4, run_label = "recruit"
)

# Check for missing
ckpts     <- list.files(OUTPUT_DIR, pattern = "checkpoint_recruit_gauss_.*\\.rds")
completed <- sort(as.integer(gsub(".*b(\\d+)\\.rds", "\\1", ckpts)))
missing   <- sort(setdiff(1:100, completed))

if (length(missing) > 0) {
  rerun_missing_gaussian(
    missing_bs = missing, data_sf = spawner_sf,
    station_locations_sf = station_locations_sf_named,
    grid_info = grid_info, spde_comps = spde_comps,
    n_cores = 1, run_label = "spawner"
  )
}