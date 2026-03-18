library(terra)
library(sf)
library(ggplot2)

# ==============================================================================
# TEMPERATURE EXTRACTION & FORMATTING
# ==============================================================================
load("ebs_bottom_temperature.rda")
my_raster <- unwrap(ebs_bottom_temperature)

station_locations_sf_named <- station_locations_sf %>%
  rename(station = GIS_STATION)

rebuilt_grid_info <- build_grid(
  survey_domain = survey_domain, 
  station_locations_sf = station_locations_sf_named, 
  cellsize = CELLSIZE
)

# 2. Mock up the 'test' object so the temperature code works without changes
test <- list(grid_info = rebuilt_grid_info)

# ==============================================================================
# TEMPERATURE EXTRACTION & GAP-FILLING (OPTION 2)
# ==============================================================================

# 1. Transform your valid grid to match the raster's CRS
valid_polys <- test$grid_info$valid_grid
valid_polys_3338 <- st_transform(valid_polys, st_crs(my_raster))
valid_polys_vect <- vect(valid_polys_3338)

cat("Spatially gap-filling missing raster data (this may take a moment)...\n")
# 2. Fill NA holes using the average of the surrounding 3x3 pixel neighborhood
my_raster_filled <- focal(my_raster, w = 3, fun = mean, NAonly = TRUE, na.rm = TRUE)

cat("Extracting filled raster values for valid grid cells...\n")
# 3. Extract using the gap-filled raster
temp_extracted <- terra::extract(my_raster_filled, valid_polys_vect, fun = mean, na.rm = TRUE)

# 4. Initialize the new array for the Vision Transformer [36, 50, 50]
target_years <- as.character(1988:2023)
n_years_new <- length(target_years)
temp_grids_new <- array(0, dim = c(n_years_new, PAD_NY, PAD_NX))

# 5. Loop through target years and fill the matrices
for (i in seq_along(target_years)) {
  yr_str <- target_years[i]
  
  if (yr_str %in% colnames(temp_extracted)) {
    cell_values <- temp_extracted[[yr_str]]
    
    # FAIL-SAFE: If the 3x3 window wasn't large enough to fill a massive hole,
    # patch any remaining NAs with the overall year's mean to prevent crashing.
    if (any(is.na(cell_values))) {
      yr_mean <- mean(cell_values, na.rm = TRUE)
      cell_values[is.na(cell_values)] <- yr_mean 
    }
    
  } else {
    cell_values <- rep(0, test$grid_info$n_valid)
  }
  
  temp_grids_new[i, , ] <- fill_matrix(cell_values, test$grid_info)
}

# 6. Apply the 2020 mask (Zero-out the missing survey year)
year_2020_idx <- which(target_years == "2020")
if (length(year_2020_idx) > 0) {
  temp_grids_new[year_2020_idx, , ] <- 0
}

# 7. Save the outputs
saveRDS(temp_grids_new, file.path(OUTPUT_DIR, "gridded_bottom_temp.rds"))
cat("Saved gridded_bottom_temp.rds\n")

if (requireNamespace("reticulate", quietly = TRUE)) {
  np <- reticulate::import("numpy")
  np$save(file.path(OUTPUT_DIR, "gridded_bottom_temp.npy"), temp_grids_new)
  cat("Saved gridded_bottom_temp.npy\n")
}
# ==============================================================================
# VISUAL VERIFICATION
# ==============================================================================

plot_temp_check <- function(temp_array, grid_info, survey_domain, target_year = "2022") {
  #' Extracts a specific year from the 3D array and maps it back to the sf polygons
  
  year_idx <- which(1988:2023 == as.numeric(target_year))
  if(length(year_idx) == 0) stop("Year not found in 1988:2023 sequence.")
  
  # Pull the 50x50 matrix for that year
  temp_matrix <- temp_array[year_idx, , ]
  
  # Extract just the valid cells back into a vector using grid_row/grid_col
  vals <- numeric(grid_info$n_valid)
  for (i in 1:grid_info$n_valid) {
    vals[i] <- temp_matrix[grid_info$grid_row[i], grid_info$grid_col[i]]
  }
  print(vals)
  
  # Map it!
  plot_sf <- st_sf(
    geometry = grid_info$valid_grid,
    temp = vals
  )
  
  
  ggplot() +
    geom_sf(data = plot_sf, aes(fill = temp), color = NA) +
    geom_sf(data = survey_domain, fill = NA, color = "red", linewidth = 1) +
    scale_fill_viridis_c(name = "Bottom Temp (°C)", option = "plasma") +
    theme_minimal() +
    labs(title = sprintf("Bottom Temperature Check — Year %s", target_year))
}

# Run the plot to test it out (using 2022 as an example)
plot_temp_check(temp_grids_new, test$grid_info, survey_domain, target_year = "1995")
