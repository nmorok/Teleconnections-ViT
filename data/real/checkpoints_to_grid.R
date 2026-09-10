# ==============================================================================
# STEP 0 — Assemble Tweedie checkpoints into gridded arrays
#
# Run this BEFORE compare_tweedie_vs_gaussian.R.
# Reads checkpoint_spawner_b001.rds ... checkpoint_recruit_b100.rds from
# CHECKPOINT_DIR and writes gridded_spawners.rds, gridded_recruits.rds,
# spatial_mask.rds, year_mask.rds, years.rds into TWEEDIE_DIR.
#
# Prerequisites in your R session (same as the main pipeline):
#   - grid_info, tw_mask (spatial_mask), spawner_sf, recruit_sf already loaded
#     OR set PAD_NY / PAD_NX / N_BOOTSTRAPS manually below and supply mask.
# ==============================================================================

# ── EDIT THESE ────────────────────────────────────────────────────────────────
CHECKPOINT_DIR <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output/Gaussian/325"
# folder containing checkpoint_spawner_b001.rds etc.
TWEEDIE_DIR    <- "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output/Gaussian/325"   # destination — will be created if needed
N_BOOTSTRAPS   <- 100
PAD_NY         <- 50
PAD_NX         <- 50

# ─────────────────────────────────────────────────────────────────────────────

dir.create(TWEEDIE_DIR, showWarnings = FALSE, recursive = TRUE)

# ── YEARS + MASKS ─────────────────────────────────────────────────────────────
# 2020 had no survey — insert as a zero slice
all_years     <- 1988:2023          # 36 years
n_years       <- length(all_years)
year_2020_idx <- which(all_years == 2020)  # 33

year_mask                <- rep(1L, n_years)
year_mask[year_2020_idx] <- 0L

# Years WITHOUT 2020 (what the checkpoints actually contain — 35 years)
survey_years  <- all_years[all_years != 2020]
n_survey_yrs  <- length(survey_years)   # 35

# ── SPATIAL MASK ──────────────────────────────────────────────────────────────
# Option A (preferred): load from grid_info if already in session
if (exists("grid_info")) {
  mask <- fill_matrix(rep(1, grid_info$n_valid), grid_info)
  cat(sprintf("[mask] Built from grid_info: %d valid cells in %dx%d\n",
              sum(mask), PAD_NY, PAD_NX))
  
  # Option B: load from an existing Gaussian output directory if available
} else if (file.exists(file.path("Gaussian", "spatial_mask.rds"))) {
  mask <- readRDS(file.path("Gaussian", "spatial_mask.rds"))
  cat(sprintf("[mask] Loaded from Gaussian/spatial_mask.rds: %d valid cells\n", sum(mask)))
  
  # Option C: all cells valid (fallback — replace with correct mask)
} else {
  warning("Could not find spatial mask — using all-ones mask. Set grid_info or supply mask manually.")
  mask <- matrix(1L, PAD_NY, PAD_NX)
}

# ── HELPER: insert 2020 zero-slice ────────────────────────────────────────────
insert_2020 <- function(raw_array) {
  # raw_array shape: [N_BOOTSTRAPS, n_survey_yrs, PAD_NY, PAD_NX]
  out <- array(0, dim = c(N_BOOTSTRAPS, n_years, PAD_NY, PAD_NX))
  out[, 1:(year_2020_idx - 1), , ]           <- raw_array[, 1:(year_2020_idx - 1), , ]
  # slot year_2020_idx stays zero
  out[, (year_2020_idx + 1):n_years, , ]     <- raw_array[, year_2020_idx:n_survey_yrs, , ]
  out
}

# ── ASSEMBLE ONE STAGE ────────────────────────────────────────────────────────
assemble_stage <- function(stage) {
  cat(sprintf("\n=== Assembling %s checkpoints ===\n", stage))
  
  raw <- array(0, dim = c(N_BOOTSTRAPS, n_survey_yrs, PAD_NY, PAD_NX))
  
  n_found   <- 0
  n_missing <- 0
  
  for (b in seq_len(N_BOOTSTRAPS)) {
    ckpt <- file.path(CHECKPOINT_DIR,
                      sprintf("checkpoint_%s_gauss_b%03d.rds", stage, b))
    if (file.exists(ckpt)) {
      raw[b, , , ] <- readRDS(ckpt)
      n_found <- n_found + 1
    } else {
      cat(sprintf("  WARNING: missing checkpoint b%03d — filled with zeros\n", b))
      n_missing <- n_missing + 1
    }
  }
  
  cat(sprintf("  Loaded: %d / %d  |  Missing (zero-filled): %d\n",
              n_found, N_BOOTSTRAPS, n_missing))
  
  out <- insert_2020(raw)
  cat(sprintf("  Final array: [%s]\n", paste(dim(out), collapse = " x ")))
  out
}

# ── RUN BOTH STAGES ───────────────────────────────────────────────────────────
spawner_grids <- assemble_stage("spawner")
recruit_grids <- assemble_stage("recruit")

# ── SAVE ──────────────────────────────────────────────────────────────────────
cat("\nSaving to", TWEEDIE_DIR, "...\n")

saveRDS(spawner_grids, file.path(TWEEDIE_DIR, "gridded_spawners.rds"))
saveRDS(recruit_grids, file.path(TWEEDIE_DIR, "gridded_recruits.rds"))
saveRDS(mask,          file.path(TWEEDIE_DIR, "spatial_mask.rds"))
saveRDS(year_mask,     file.path(TWEEDIE_DIR, "year_mask.rds"))
saveRDS(all_years,     file.path(TWEEDIE_DIR, "years.rds"))

cat("Done. Files written:\n")
for (f in c("gridded_spawners.rds", "gridded_recruits.rds",
            "spatial_mask.rds", "year_mask.rds", "years.rds"))
  cat(sprintf("  %s/%s\n", TWEEDIE_DIR, f))

cat("\nNow update TWEEDIE_DIR in compare_tweedie_vs_gaussian.R to: '", TWEEDIE_DIR, "'\n", sep = "")
