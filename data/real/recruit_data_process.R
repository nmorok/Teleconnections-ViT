# ==============================================================================
# RECRUIT BOOTSTRAP GRIDDING — standalone script
#
# Assumes spawner checkpoints are already on disk from the previous run.
# Does NOT rerun spawners. Loads them at the end for final assembly only.
#
# Prerequisites in your R session:
#   - All functions from spde_gridding_tweedie_fixedp.R sourced
#   - recruit_sf, spawner_sf, station_locations_sf_named,
#     survey_domain, sf_maps already loaded
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: Rebuild grid + mesh (same parameters as spawner run — must match)
# ------------------------------------------------------------------------------

cat("=== Recruit Bootstrap Gridding ===\n\n")

cat("[1/5] Rebuilding grid and mesh...\n")
grid_info  <- build_grid(survey_domain, station_locations_sf_named, cellsize = CELLSIZE)
stn_coords <- st_coordinates(station_locations_sf_named)
spde_comps <- build_spde(stn_coords, grid_info)


# ------------------------------------------------------------------------------
# STEP 2: Estimate recruit p by year
# Saves to tweedie_p_by_year_recruit.rds — won't overwrite spawner's file
# ------------------------------------------------------------------------------

recruit_p_file <- file.path(OUTPUT_DIR, "tweedie_p_by_year_recruit.rds")

if (file.exists(recruit_p_file)) {
  cat("\n[2/5] Loading existing recruit p estimates...\n")
  recruit_p_by_year <- readRDS(recruit_p_file)
  print(recruit_p_by_year)
} else {
  cat("\n[2/5] Estimating Tweedie p by year for recruits...\n")
  recruit_p_by_year <- estimate_tweedie_p_by_year(
    data_sf              = recruit_sf,
    station_locations_sf = station_locations_sf_named,
    spde_comps           = spde_comps,
    timeout_sec          = 1800,
    output_dir           = OUTPUT_DIR
  )
  # Save under recruit-specific filename
  saveRDS(recruit_p_by_year, recruit_p_file)
  write.csv(
    data.frame(year = as.integer(names(recruit_p_by_year)),
               p    = as.numeric(recruit_p_by_year)),
    file.path(OUTPUT_DIR, "tweedie_p_by_year_recruit.csv"),
    row.names = FALSE
  )
  cat("  Saved recruit p to tweedie_p_by_year_recruit.{rds,csv}\n")
}


# ------------------------------------------------------------------------------
# STEP 3: Run recruit bootstraps
# Uses seed = SEED + 1000 to match the original pipeline
# Checkpoints saved as checkpoint_recruit_b001.rds etc.
# ------------------------------------------------------------------------------

cat("\n[3/5] Running recruit bootstraps...\n")

recruit_result <- run_bootstrap_parallel(
  data_sf              = recruit_sf,
  station_locations_sf = station_locations_sf_named,
  grid_info            = grid_info,
  spde_comps           = spde_comps,
  p_by_year            = recruit_p_by_year,
  n_bootstraps         = N_BOOTSTRAPS,
  n_subsample          = N_SUBSAMPLE,
  seed                 = SEED,
  n_cores              = 4,
  run_label            = "recruit"
)


# ------------------------------------------------------------------------------
# STEP 4: Check for and rerun any missing recruit bootstraps
# ------------------------------------------------------------------------------

cat("\n[4/5] Checking recruit checkpoints...\n")

recruit_ckpts     <- list.files(OUTPUT_DIR, pattern = "checkpoint_recruit_b.*\\.rds")
recruit_completed <- sort(as.integer(gsub(".*b(\\d+)\\.rds", "\\1", recruit_ckpts)))
recruit_missing   <- sort(setdiff(1:N_BOOTSTRAPS, recruit_completed))

cat(sprintf("  Completed: %d / %d\n", length(recruit_completed), N_BOOTSTRAPS))

if (length(recruit_missing) > 0) {
  cat(sprintf("  Missing: %s\n", paste(recruit_missing, collapse = ", ")))
  cat("  Rerunning missing recruit bootstraps...\n")
  
  rerun_missing(
    missing_bs           = recruit_missing,
    data_sf              = recruit_sf,
    station_locations_sf = station_locations_sf_named,
    grid_info            = grid_info,
    spde_comps           = spde_comps,
    p_by_year            = recruit_p_by_year,
    seed                 = SEED,
    n_cores              = 1,
    run_label            = "recruit"
  )
} else {
  cat("  All recruit bootstraps complete.\n")
}

recruit_p_by_year <- readRDS(file.path(OUTPUT_DIR, "tweedie_p_by_year_recruit.rds"))

rerun_missing(
  missing_bs           = 63,
  data_sf              = recruit_sf,
  station_locations_sf = station_locations_sf_named,
  grid_info            = test$grid_info,
  spde_comps           = test$spde_comps,
  p_by_year            = recruit_p_by_year,
  seed                 = SEED,   # must match original recruit seed
  run_label            = "recruit"
)

# ------------------------------------------------------------------------------
# STEP 5: Assemble final arrays from checkpoints (spawner + recruit)
# Loads spawner checkpoints from disk — no rerunning
# Inserts 2020 as zeros for both, saves all outputs
# ------------------------------------------------------------------------------

cat("\n[5/5] Assembling final outputs...\n")

years_spawner <- sort(unique(spawner_sf$year))
years_recruit <- sort(unique(recruit_sf$year))
n_years_s     <- length(years_spawner)
n_years_r     <- length(years_recruit)

# Load spawner checkpoints
cat("  Loading spawner checkpoints from disk...\n")
spawner_raw <- array(0, dim = c(N_BOOTSTRAPS, n_years_s, PAD_NY, PAD_NX))
for (b in 1:N_BOOTSTRAPS) {
  ckpt <- file.path(OUTPUT_DIR, sprintf("checkpoint_spawner_b%03d.rds", b))
  if (file.exists(ckpt)) {
    spawner_raw[b, , , ] <- readRDS(ckpt)
  } else {
    cat(sprintf("  WARNING: spawner checkpoint b%03d missing — filled zeros\n", b))
  }
}

# Load recruit checkpoints
cat("  Loading recruit checkpoints from disk...\n")
recruit_raw <- array(0, dim = c(N_BOOTSTRAPS, n_years_r, PAD_NY, PAD_NX))
for (b in 1:N_BOOTSTRAPS) {
  ckpt <- file.path(OUTPUT_DIR, sprintf("checkpoint_recruit_b%03d.rds", b))
  if (file.exists(ckpt)) {
    recruit_raw[b, , , ] <- readRDS(ckpt)
  } else {
    cat(sprintf("  WARNING: recruit checkpoint b%03d missing — filled zeros\n", b))
  }
}

# Insert 2020 as zeros (no survey that year)
all_years     <- 1988:2023
n_years_new   <- length(all_years)   # 36
year_2020_idx <- which(all_years == 2020)  # 33

insert_2020 <- function(raw_array, n_years_orig) {
  n_boot    <- dim(raw_array)[1]
  out       <- array(0, dim = c(n_boot, n_years_new, PAD_NY, PAD_NX))
  out[, 1:(year_2020_idx - 1), , ]          <- raw_array[, 1:(year_2020_idx - 1), , ]
  # slot year_2020_idx stays zero
  out[, (year_2020_idx + 1):n_years_new, ,] <- raw_array[, year_2020_idx:n_years_orig, , ]
  return(out)
}

spawner_grids_new <- insert_2020(spawner_raw, n_years_s)
recruit_grids_new <- insert_2020(recruit_raw, n_years_r)

year_mask                <- rep(1L, n_years_new)
year_mask[year_2020_idx] <- 0L

mask <- fill_matrix(rep(1, grid_info$n_valid), grid_info)

# Save RDS
saveRDS(spawner_grids_new, file.path(OUTPUT_DIR, "gridded_spawners.rds"))
saveRDS(recruit_grids_new, file.path(OUTPUT_DIR, "gridded_recruits.rds"))
saveRDS(mask,              file.path(OUTPUT_DIR, "spatial_mask.rds"))
saveRDS(year_mask,         file.path(OUTPUT_DIR, "year_mask.rds"))
saveRDS(all_years,         file.path(OUTPUT_DIR, "years.rds"))

# Save NPY
if (requireNamespace("reticulate", quietly = TRUE)) {
  np <- reticulate::import("numpy")
  np$save(file.path(OUTPUT_DIR, "gridded_spawners.npy"), spawner_grids_new)
  np$save(file.path(OUTPUT_DIR, "gridded_recruits.npy"), recruit_grids_new)
  np$save(file.path(OUTPUT_DIR, "spatial_mask.npy"),     mask)
  np$save(file.path(OUTPUT_DIR, "year_mask.npy"),        year_mask)
  np$save(file.path(OUTPUT_DIR, "years.npy"),            as.integer(all_years))
  cat("  Saved .npy files\n")
}

cat(sprintf("\nDone. Final arrays:\n"))
cat(sprintf("  Spawners: [%s]\n", paste(dim(spawner_grids_new), collapse = " x ")))
cat(sprintf("  Recruits: [%s]\n", paste(dim(recruit_grids_new), collapse = " x ")))
cat(sprintf("  Years:    %d-%d (%d total, 2020 masked)\n",
            min(all_years), max(all_years), n_years_new))
cat(sprintf("  Mask:     %d valid cells in %dx%d grid\n",
            sum(mask), PAD_NY, PAD_NX))