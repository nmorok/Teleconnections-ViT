# ================================================================
# Eastern Bering Sea: Publication Figures 1–6
# ================================================================
# Figure 1 – EBS NOAA survey extent overlaid on a regular grid
# Figure 2 – 349 survey station points (grey)
# Figure 3 – 300 of those points highlighted red, 49 remain grey
# Figure 4 – 2×2 panel: GMRF heatmap clipped to survey extent
#             with 300 red points; each panel is a new realization
# Figure 5 – 2×2 panel: integrated-gradients-style hotspot map;
#             sparse short-range GP → most cells grey, hotspots colored
# Figure 6 – MAE time series: observed (deep purple) vs.
#             predicted (bright orange) recruitment, with MAE annotation
# ================================================================

# ── 0. Install / load packages ─────────────────────────────────
# akgfmaps lives on GitHub (AFSC GAP Products group):
#   remotes::install_github("afsc-gap-products/akgfmaps",
#                           build_vignettes = FALSE)
# All other packages are on CRAN.

pkgs <- c("akgfmaps", "ggplot2", "sf", "viridis", "patchwork", "MASS", "ggspatial")
invisible(lapply(pkgs, library, character.only = TRUE))

#set.seed(42)

# ── 1. EBS survey layers (EPSG:3338 = Alaska Albers, metres) ───
ebs         <- akgfmaps::get_base_layers(select.region = "ebs",
                                          set.crs      = "EPSG:3338")
survey_area <-ebs$survey.area[ebs$survey.area$SURVEY_NAME == "Eastern Bering Crab/Groundfish Bottom Trawl Survey",]
 #ebs$survey.area#[ebs$survey.area$survey == "ebs" , ]
land        <- ebs$akland[ebs$akland$COUNTRY == "US", ]   # row-subset: trailing comma required for sf objects

# ── 2. Regular 40×40 grid over survey bounding box ─────────────
grid_cells <- sf::st_make_grid(survey_area, n = c(40, 40), square = TRUE)
grid_sf    <- sf::st_sf(cell_id  = seq_along(grid_cells),
                         geometry = grid_cells)

survey_union <- sf::st_union(survey_area)
is_inside    <- lengths(sf::st_intersects(grid_sf, survey_union)) > 0
grid_inside  <- grid_sf[ is_inside, ]
grid_outside <- grid_sf[!is_inside, ]

# ── 3. Survey station coordinates from CSV (decimal degrees) ───
# station_latlon.csv contains one row per unique station (Lat, Long in WGS84).
coords_raw <- read.csv("station_latlon.csv")
pts_sf <- sf::st_as_sf(coords_raw,
                        coords = c("Long", "Lat"),
                        crs    = 4326) |>
  sf::st_transform(crs = "EPSG:3338")
pts_sf$pt_id <- seq_len(nrow(pts_sf))

sel      <- sort(sample(nrow(pts_sf), 300))
pts_red  <- pts_sf[ sel, ]
pts_grey <- pts_sf[-sel, ]

# ── 4. GP / GMRF simulation on interior grid-cell centroids ────
#   Uses an isotropic exponential covariance via Cholesky.
#   phi controls the spatial range (metres).

cents <- sf::st_coordinates(sf::st_centroid(grid_inside))
nc    <- nrow(cents)

bb  <- sf::st_bbox(survey_area)
phi <- as.numeric(bb["xmax"] - bb["xmin"]) * 0.25   # ~25% of x-extent

D     <- as.matrix(dist(cents))
Sigma <- exp(-D / phi) + 1e-6 * diag(nc)  # nugget for numerical stability
L     <- chol(Sigma)                        # upper Cholesky: t(L) %*% L = Sigma

sim_raw  <- replicate(4, as.numeric(t(L) %*% rnorm(nc)))

# Normalise to a shared [0, 1] scale across all 4 panels
lo       <- min(sim_raw); hi <- max(sim_raw)
sim_norm <- (sim_raw - lo) / (hi - lo)
# Pre-build one sf object per panel — keeps grid_inside and sim_norm in sync
# and avoids $<- dispatch issues on tbl_sf / sf objects.
stopifnot(nrow(grid_inside) == nc)            # fast-fail if environment drifted
gd_list <- lapply(1:4, function(k) {
  sf::st_sf(
    cell_id  = grid_inside$cell_id,
    value    = sim_norm[, k],
    geometry = sf::st_geometry(grid_inside)
  )
})

# 
# ── 5. Map extent & shared publication theme ────────────────────
# Derive limits from the grid itself — this is the domain we want all
# panels to share.  clip = "on" hard-clips points/geometries at the border.
grid_bb  <- sf::st_bbox(grid_sf)
xlim_map <- c(grid_bb["xmin"], grid_bb["xmax"])
ylim_map <- c(grid_bb["ymin"], grid_bb["ymax"])

pub_theme <- theme_bw(base_size = 9) +
  theme(
    panel.grid        = element_blank(),
    axis.title        = element_blank(),
    axis.text         = element_text(size = 7, color = "grey25"),
    axis.ticks        = element_line(linewidth = 0.3),
    panel.border      = element_rect(linewidth = 0.5, color = "grey30"),
    plot.title        = element_text(size = 10, face = "bold", hjust = 0),
    plot.margin       = margin(4, 4, 4, 4, "pt"),
    legend.key.height = unit(1.2, "cm"),
    legend.key.width  = unit(0.33, "cm"),
    legend.text       = element_text(size = 7),
    legend.title      = element_text(size = 8, face = "bold"),
    legend.background = element_blank()
  )

# Reusable layer list for Figures 1–3
base_map <- list(
 # geom_sf(data = grid_sf,     fill = NA,           color = "grey95",
         # linewidth = 0.18),
  geom_sf(data = survey_area, fill = "steelblue",   alpha = 0.11,
          color = "steelblue",    linewidth = 0.50),
  geom_sf(data = land,        fill = "grey60",      color = "grey80",
          linewidth = 0.15),
  coord_sf(xlim = xlim_map, ylim = ylim_map, expand = FALSE, clip = "on")
)

# ── 6. Figure 1: EBS inset map ─────────────────────────────────-
# Annotation anchor: land visible within map extent (so text stays on-screen)
land_visible <- sf::st_crop(land, sf::st_bbox(grid_sf))
alaska_pt    <- sf::st_coordinates(
  sf::st_point_on_surface(sf::st_union(land_visible))
)
ebs_pt <- sf::st_coordinates(sf::st_centroid(sf::st_union(survey_area)))

# Inset theme: larger axis text, clear graticule labels
fig1_theme <- pub_theme +
  theme(axis.text = element_text(size = 10, color = "grey10"))

fig1 <- ggplot() +
  geom_sf(data = survey_area, fill = "steelblue", alpha = 0.22,
          color = "steelblue", linewidth = 0.60) +
  geom_sf(data = land, fill = "grey60", color = "grey75",
          linewidth = 0.15) +
  annotation_north_arrow(
    location     = "tl", which_north = "true",
    style        = north_arrow_orienteering(
      fill       = c("grey20", "white"),
      line_col   = "grey20", text_col = "grey20", text_size = 8
    ),
    height = unit(1.1, "cm"), width  = unit(1.1, "cm"),
    pad_x  = unit(0.35, "cm"), pad_y = unit(0.35, "cm")
  ) +
  annotate("text",
           x = alaska_pt[1, "X"], y = alaska_pt[1, "Y"],
           label    = "Alaska",
           fontface = "italic", size = 4, color = "grey20") +
  annotate("text",
           x = ebs_pt[1, "X"], y = ebs_pt[1, "Y"],
           label      = "Eastern Bering Sea\nCrab Survey",
           fontface   = "italic", size  = 3.2,
           color      = "steelblue4", lineheight = 0.9) +
  # coord_sf(
  #   xlim  = xlim_map, ylim = ylim_map, expand = FALSE, clip = "on",
  #   datum = sf::st_crs(4326)   # display axis labels as lat / lon degrees
  # ) +
  fig1_theme +
  labs(title = "A")

# ── 7. Figure 2: + all 350 simulated points ─────────────────────
fig2 <- ggplot() +
    geom_sf(data = grid_sf,     fill = NA,           color = "grey95",
          linewidth = 0.18) +
  geom_sf(data = pts_sf, color = "grey30", fill = 'grey30', size = 3,
          alpha = 0.65) +
  pub_theme +
 base_map +
  labs(title = "B")

# ── 8. Figure 3: 300 red + 50 remaining grey ────────────────────
fig3 <- ggplot() +
    geom_sf(data = grid_sf,     fill = NA,           color = "grey95",
          linewidth = 0.18) +
  geom_sf(data = pts_grey, color = "grey30",  size = 2,  alpha = 0.65,
          shape = 16) +
  geom_sf(data = pts_red,  color = "#C0392B", fill = 'grey30',
   size = 3, alpha = 1,
          shape = 16) +
  pub_theme +
 base_map +
  labs(title = "C")

# ── 9. Figure 4: 2×2 GMRF panel ─────────────────────────────────
#   Outside survey extent: grey cells.
#   Inside survey extent: viridis-plasma fill from GMRF simulation.
#   Red points fixed across panels; 50 grey points removed.

gmrf_panel <- function(k) {
  gd <- gd_list[[k]]          # pre-built sf with value column attached

  ggplot() +
    geom_sf(data = grid_outside, fill = "grey88", color = "grey78",
            linewidth = 0.15) +
    geom_sf(data = gd,           aes(fill = value), color = NA) +
    geom_sf(data = survey_area,  fill = NA, color = "black",
            linewidth = 0.45) +
    geom_sf(data = land,         fill = "grey52", color = "grey42",
            linewidth = 0.15) +
    geom_sf(data = pts_red, color = "#C0392B", size = 0.85,
            alpha = 0.85, shape = 16) +
    scale_fill_viridis_c(
      option = "plasma", direction = -1, end = 0.95,
      limits = c(0, 1),
      name   = "Spawner or Recruit density via INLA",
      breaks = c(0, 0.5, 1),
      labels = c("Low", "Mid", "High")
    ) +
    coord_sf(xlim = xlim_map, ylim = ylim_map, expand = FALSE, clip = "on") +
    pub_theme +
    # legend.position must be set per-panel so patchwork can collect it;
    # using & after plot_annotation() breaks in current patchwork versions
    theme(legend.position = "right") +
    labs(title = paste0("Realization ", k))
}

fig4 <- wrap_plots(lapply(1:4, gmrf_panel), ncol = 2) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "D",
    theme = theme(plot.title = element_text(size = 10, face = "bold",
                                             hjust = 0))
  )

# ── 10. Figure 5: Integrated-gradients-style hotspot map ────────
# Short-range GP → absolute value → sparse thresholding:
# bottom 75 % of cells masked to grey (NA); top 25 % get color.

phi_ig   <- as.numeric(bb["xmax"] - bb["xmin"]) * 0.07  # short range → spotty
Sigma_ig <- exp(-D / phi_ig) + 1e-6 * diag(nc)
L_ig     <- chol(Sigma_ig)

ig_raw  <- replicate(4, abs(as.numeric(t(L_ig) %*% rnorm(nc))))

# Single threshold across all panels so panels share the same scale
thresh   <- quantile(ig_raw, 0.75)
ig_masked          <- ig_raw
ig_masked[ig_masked < thresh] <- NA

ig_lo   <- min(ig_masked, na.rm = TRUE)
ig_hi   <- max(ig_masked, na.rm = TRUE)
ig_norm <- (ig_masked - ig_lo) / (ig_hi - ig_lo)

ig_list <- lapply(1:4, function(k) {
  sf::st_sf(
    cell_id  = grid_inside$cell_id,
    value    = ig_norm[, k],
    geometry = sf::st_geometry(grid_inside)
  )
})

ig_panel <- function(k) {
  gd <- ig_list[[k]]

  ggplot() +
    geom_sf(data = grid_outside, fill = "grey88", color = "grey78",
            linewidth = 0.15) +
    geom_sf(data = gd,           aes(fill = value), color = NA) +
    geom_sf(data = survey_area,  fill = NA, color = "black",
            linewidth = 0.45) +
    geom_sf(data = land,         fill = "grey52", color = "grey42",
            linewidth = 0.15) +
    scale_fill_viridis_c(
      option   = "inferno",
      na.value = "grey82",
      limits   = c(0, 1),
      name     = "Attribution\n(IG score)",
      breaks   = c(0, 0.5, 1),
      labels   = c("Low", "Mid", "High")
    ) +
    coord_sf(xlim = xlim_map, ylim = ylim_map, expand = FALSE, clip = "on") +
    pub_theme +
    theme(legend.position = "right") +
    labs(title = paste0("Realization ", k))
}

fig5 <- wrap_plots(lapply(1:4, ig_panel), ncol = 2) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "E",
    theme = theme(plot.title = element_text(size = 10, face = "bold", hjust = 0))
  )

# ── 11. Figure 6: Recruitment time-series with MAE annotation ───
set.seed(7)
years   <- 1990:2022
n_yr    <- length(years)

# Observed: smooth sine trend + mild random walk, scaled to [0, 1]
trend    <- sin(seq(0, 2 * pi, length.out = n_yr)) * 0.35
obs_rec  <- 1 + trend + cumsum(rnorm(n_yr, 0, 0.08)) * 0.4
obs_rec  <- (obs_rec - min(obs_rec)) / diff(range(obs_rec))

# Predicted: closely tracks observed with small Gaussian noise
pred_rec <- pmax(0, obs_rec + rnorm(n_yr, 0, 0.038))

mae_val  <- mean(abs(obs_rec - pred_rec))

ts_df <- data.frame(
  Year        = rep(years, 2),
  Recruitment = c(obs_rec, pred_rec),
  Series      = rep(c("Observed", "Predicted"), each = n_yr)
)

fig6 <- ggplot(ts_df, aes(x = Year, y = Recruitment, color = Series)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 1.8) +
  scale_color_manual(
    values = c("Observed" = "#3B0764", "Predicted" = "#FF6D00"),
    name   = NULL
  ) +
  annotate("label",
           x = max(years) - 1, y = max(ts_df$Recruitment) * 0.96,
           label         = sprintf("MAE = %.3f", mae_val),
           hjust         = 1, size = 3.5,
           fill          = "white", color = "grey20",
           label.padding = unit(0.35, "lines")) +
  scale_x_continuous(breaks = seq(1990, 2022, by = 5)) +
  labs(title = "F", y = "Recruitment index", x = "Year") +
  theme_bw(base_size = 9) +
  theme(
    panel.grid.minor  = element_blank(),
    panel.grid.major  = element_line(linewidth = 0.25, color = "grey88"),
    axis.title        = element_text(size = 8, color = "grey20"),
    axis.text         = element_text(size = 7, color = "grey25"),
    panel.border      = element_rect(linewidth = 0.5, color = "grey30"),
    plot.title        = element_text(size = 10, face = "bold", hjust = 0),
    plot.margin       = margin(4, 8, 4, 4, "pt"),
    legend.position   = c(0.13, 0.88),
    legend.background = element_rect(fill = "white", linewidth = 0.3,
                                     color = "grey70"),
    legend.key.size   = unit(0.55, "cm"),
    legend.text       = element_text(size = 8)
  )

# ── 12. Export ──────────────────────────────────────────────────
ggsave("fig1_survey_grid.png",     fig1, width = 6,  height = 5.5,
       dpi = 300, bg = "white")
ggsave("fig2_all_points.png",      fig2, width = 6,  height = 5.5,
       dpi = 300, bg = "white")
ggsave("fig3_selected_points.png", fig3, width = 6,  height = 5.5,
       dpi = 300, bg = "white")
ggsave("fig4_gmrf_panels.png",     fig4, width = 11, height = 9.5,
       dpi = 300, bg = "white")
ggsave("fig5_hotspot_ig.png",      fig5, width = 11, height = 9.5,
       dpi = 300, bg = "white")
ggsave("fig6_mae_timeseries.png",  fig6, width = 7,  height = 4,
       dpi = 300, bg = "white")

message("Done — figures saved to: ", normalizePath(getwd()))
