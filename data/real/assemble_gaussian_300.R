# ==============================================================================
# Assemble Gaussian/300 checkpoints (100 bootstraps x 35 survey years x 50x50)
# into gridded_spawners/recruits arrays with 2020 zero-inserted (36 years),
# and write both .rds and .npy outputs to:
#   1) data/real/output/pub/Gaussian/300/   (archival copy next to checkpoints)
#   2) data/real/output/                    (what create_splits.py reads)
#
# Reuses the existing spatial_mask.rds from data/real/output/ since the
# survey grid/domain does not change with N_SUBSAMPLE.
# ==============================================================================

CHECKPOINT_DIR <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output/pub/Gaussian/300"
ARCHIVE_DIR    <- CHECKPOINT_DIR
FINAL_DIR      <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output"
N_BOOTSTRAPS   <- 100
PAD_NY         <- 50
PAD_NX         <- 50

all_years     <- 1988:2023
n_years       <- length(all_years)
year_2020_idx <- which(all_years == 2020)      # 33

survey_years <- all_years[all_years != 2020]   # 35 years, matches checkpoints
n_survey_yrs <- length(survey_years)

year_mask                <- rep(1L, n_years)
year_mask[year_2020_idx] <- 0L

mask <- readRDS(file.path(FINAL_DIR, "spatial_mask.rds"))
cat(sprintf("[mask] Loaded from %s/spatial_mask.rds: %d valid cells\n",
            FINAL_DIR, sum(mask)))

insert_2020 <- function(raw_array) {
  out <- array(0, dim = c(N_BOOTSTRAPS, n_years, PAD_NY, PAD_NX))
  out[, 1:(year_2020_idx - 1), , ]       <- raw_array[, 1:(year_2020_idx - 1), , ]
  out[, (year_2020_idx + 1):n_years, , ] <- raw_array[, year_2020_idx:n_survey_yrs, , ]
  out
}

assemble_stage <- function(stage) {
  cat(sprintf("\n=== Assembling %s checkpoints ===\n", stage))
  raw <- array(0, dim = c(N_BOOTSTRAPS, n_survey_yrs, PAD_NY, PAD_NX))
  n_found <- 0
  for (b in seq_len(N_BOOTSTRAPS)) {
    ckpt <- file.path(CHECKPOINT_DIR, sprintf("checkpoint_%s_gauss_b%03d.rds", stage, b))
    if (file.exists(ckpt)) {
      raw[b, , , ] <- readRDS(ckpt)
      n_found <- n_found + 1
    } else {
      cat(sprintf("  WARNING: missing checkpoint b%03d — filled with zeros\n", b))
    }
  }
  cat(sprintf("  Loaded: %d / %d\n", n_found, N_BOOTSTRAPS))
  out <- insert_2020(raw)
  cat(sprintf("  Final array: [%s]\n", paste(dim(out), collapse = " x ")))
  out
}

spawner_grids <- assemble_stage("spawner")
recruit_grids <- assemble_stage("recruit")

write_all <- function(dir) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  saveRDS(spawner_grids, file.path(dir, "gridded_spawners.rds"))
  saveRDS(recruit_grids, file.path(dir, "gridded_recruits.rds"))
  saveRDS(mask,          file.path(dir, "spatial_mask.rds"))
  saveRDS(year_mask,     file.path(dir, "year_mask.rds"))
  saveRDS(all_years,     file.path(dir, "years.rds"))

  np <- reticulate::import("numpy")
  np$save(file.path(dir, "gridded_spawners.npy"), spawner_grids)
  np$save(file.path(dir, "gridded_recruits.npy"), recruit_grids)
  np$save(file.path(dir, "spatial_mask.npy"),     mask)
  np$save(file.path(dir, "year_mask.npy"),        as.integer(year_mask))
  np$save(file.path(dir, "years.npy"),            as.integer(all_years))
  cat(sprintf("Wrote gridded_spawners/recruits, spatial_mask, year_mask, years (.rds + .npy) to %s\n", dir))
}

write_all(ARCHIVE_DIR)
write_all(FINAL_DIR)

# Refresh grid_metadata.json n_bootstraps (keep other fields as-is)
library(jsonlite)
meta_path <- file.path(FINAL_DIR, "grid_metadata.json")
meta <- fromJSON(meta_path)
meta$n_bootstraps <- N_BOOTSTRAPS
write_json(meta, meta_path, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("Updated %s: n_bootstraps = %d\n", meta_path, N_BOOTSTRAPS))

cat("\nDone.\n")
