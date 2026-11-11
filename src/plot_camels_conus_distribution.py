#!/usr/bin/env python3
"""Plot publication-quality geographic distribution of 671 CAMELS basins across CONUS.

Inputs:
  - experiment/figures/fig07_spatial_data.csv (671 basins with gauge_id, lat, lon)
  - experiment/camels_attributes_v2.0/camels_clim.txt (climate attributes: p_mean, aridity, frac_snow)
  - us_states.geojson (cached CONUS state boundaries)

Outputs saved to:
  - experiment/other figures/
  - (and mirrored to ../Other Figures/ if present)

Outputs:
  1. fig02_camels_conus_distribution.png (Primary high-res study area map with aridity gradient & stats)
  2. fig02_camels_conus_precipitation.png (Distribution colored by Mean Annual Precipitation)
  3. fig01_camels_conus_hydroclimatic_4panel.png (4-panel comprehensive hydroclimatic characterization)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import urllib.request

# OpenMP runtime guard for Windows
if sys.platform.startswith("win") or os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from paths import DATA_DIR, FIGURES_DIR, MPL_CONFIG_DIR, PROJECT_ROOT

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import geopandas as gpd

# Directories
FIG_DATA_PATH = DATA_DIR / "gauge_ids.csv"
CLIM_PATH = DATA_DIR / "camels_attributes_v2.0" / "camels_clim.txt"
GEOJSON_PATH = DATA_DIR / "us_states.geojson"
OUTPUT_DIR = FIGURES_DIR / "other"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_OTHER_DIR = PROJECT_ROOT.parent / "Other Figures"

# Journal-grade styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Calibri"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def ensure_us_states_geojson() -> gpd.GeoDataFrame:
    """Download and cache CONUS state boundaries in EPSG:5070 Albers projection."""
    if not GEOJSON_PATH.exists():
        url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
        print(f"Downloading US states boundary geojson from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            GEOJSON_PATH.write_bytes(resp.read())

    gdf = gpd.read_file(GEOJSON_PATH)
    # Filter to CONUS (exclude Alaska, Hawaii, Puerto Rico)
    conus = gdf[~gdf["name"].isin(["Alaska", "Hawaii", "Puerto Rico"])].copy()
    conus = conus.to_crs(epsg=5070)
    return conus


def load_merged_data() -> gpd.GeoDataFrame:
    """Load spatial data and merge with climate attributes, projected to EPSG:5070."""
    if not FIG_DATA_PATH.exists():
        raise FileNotFoundError(f"Spatial data not found: {FIG_DATA_PATH}")

    df = pd.read_csv(FIG_DATA_PATH)
    df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(8)

    # Merge with climate attributes if available
    if CLIM_PATH.exists():
        clim = pd.read_csv(CLIM_PATH, sep=";", dtype={"gauge_id": str})
        clim["gauge_id"] = clim["gauge_id"].astype(str).str.zfill(8)
        df = df.merge(clim, on="gauge_id", how="left")
        # Calculate mean annual precipitation in mm/year
        if "p_mean" in df.columns:
            df["p_annual_mm"] = df["p_mean"] * 365.25

    # Create GeoDataFrame in WGS84 and reproject to Albers EPSG:5070
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["gauge_lon"], df["gauge_lat"]),
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs(epsg=5070)
    return gdf


def plot_primary_conus_distribution(conus: gpd.GeoDataFrame, basins: gpd.GeoDataFrame):
    """Plot primary publication-quality CONUS distribution map (Figure 2)."""
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # Base CONUS land polygon & state borders
    conus_boundary = conus.dissolve()
    conus_boundary.plot(ax=ax, color="#f8fafc", edgecolor="none", zorder=1)
    conus.plot(ax=ax, color="none", edgecolor="#cbd5e1", linewidth=0.6, zorder=2)
    conus_boundary.plot(ax=ax, color="none", edgecolor="#475569", linewidth=1.1, zorder=3)

    # Color by aridity index if present, otherwise distinctive primary color
    if "aridity" in basins.columns:
        sc = ax.scatter(
            basins.geometry.x,
            basins.geometry.y,
            c=basins["aridity"],
            cmap="YlGnBu_r",
            vmin=0.2,
            vmax=2.5,
            s=26,
            alpha=0.90,
            edgecolors="#1e293b",
            linewidths=0.45,
            zorder=4,
        )
        # Elegant horizontal colorbar in bottom-left Gulf / Mexico area
        cax = inset_axes(
            ax,
            width="34%",
            height="4.5%",
            loc="lower left",
            bbox_to_anchor=(0.04, 0.07, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.set_label("Aridity Index (PET / P)", fontsize=9, labelpad=4, fontweight="medium")
        cbar.ax.tick_params(labelsize=8)
    else:
        ax.scatter(
            basins.geometry.x,
            basins.geometry.y,
            color="#0284c7",
            s=28,
            alpha=0.88,
            edgecolors="#0f172a",
            linewidths=0.5,
            zorder=4,
            label=f"CAMELS Gauges (N = {len(basins)})",
        )
        ax.legend(loc="lower left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#cbd5e1")

    # Informative statistics card in top-right
    stats_box = (
        "Dataset: CAMELS (USGS Daymet)\n"
        f"Total Catchments: {len(basins):,} basins\n"
        "Period: 1980–2010 (30 Water Years)\n"
        "Coverage: Contiguous United States\n"
        "Input Dim: 40 (5 Dynamic + 35 Static)"
    )
    ax.text(
        0.97,
        0.95,
        stats_box,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.93, linewidth=0.8),
        zorder=5,
        linespacing=1.35,
    )

    # Compass / North Arrow
    ax.annotate(
        "N",
        xy=(0.04, 0.90),
        xytext=(0.04, 0.83),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(facecolor="#1e293b", edgecolor="#1e293b", width=1.8, headwidth=6),
        zorder=5,
    )

    # Title & Subtitle
    ax.set_title(
        "Geographic Distribution of 671 CAMELS Basins across CONUS",
        fontsize=12,
        fontweight="bold",
        pad=10,
        color="#0f172a",
    )

    # Clean styling: equal aspect, remove frame axes ticks
    ax.set_aspect("equal")
    ax.axis("off")

    # Set bounds to tightly frame CONUS
    bounds = conus.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.02
    pad_y = (bounds[3] - bounds[1]) * 0.03
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    # Save to primary and mirror output dirs
    out_file1 = OUTPUT_DIR / "fig02_camels_conus_distribution.png"
    fig.savefig(out_file1, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_file1}")

    if ROOT_OTHER_DIR.exists():
        out_file_mirror = ROOT_OTHER_DIR / "fig02_camels_conus_distribution.png"
        fig.savefig(out_file_mirror, dpi=300, bbox_inches="tight")
        print(f"Mirrored: {out_file_mirror}")

    plt.close(fig)


def plot_precipitation_distribution(conus: gpd.GeoDataFrame, basins: gpd.GeoDataFrame):
    """Plot CONUS distribution colored by Mean Annual Precipitation (mm/yr)."""
    if "p_annual_mm" not in basins.columns:
        return

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    conus_boundary = conus.dissolve()
    conus_boundary.plot(ax=ax, color="#f8fafc", edgecolor="none", zorder=1)
    conus.plot(ax=ax, color="none", edgecolor="#cbd5e1", linewidth=0.6, zorder=2)
    conus_boundary.plot(ax=ax, color="none", edgecolor="#475569", linewidth=1.1, zorder=3)

    sc = ax.scatter(
        basins.geometry.x,
        basins.geometry.y,
        c=basins["p_annual_mm"],
        cmap="Blues",
        norm=Normalize(vmin=200, vmax=2400),
        s=26,
        alpha=0.90,
        edgecolors="#1e293b",
        linewidths=0.45,
        zorder=4,
    )

    cax = inset_axes(
        ax,
        width="34%",
        height="4.5%",
        loc="lower left",
        bbox_to_anchor=(0.04, 0.07, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("Mean Annual Precipitation (mm yr⁻¹)", fontsize=9, labelpad=4, fontweight="medium")
    cbar.ax.tick_params(labelsize=8)

    ax.text(
        0.97,
        0.95,
        "Precipitation Gradient\n"
        f"Mean: {basins['p_annual_mm'].mean():.0f} mm yr⁻¹\n"
        f"Min:  {basins['p_annual_mm'].min():.0f} mm yr⁻¹\n"
        f"Max:  {basins['p_annual_mm'].max():.0f} mm yr⁻¹\n"
        f"Total Basins: {len(basins)}",
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.93, linewidth=0.8),
        zorder=5,
        linespacing=1.35,
    )

    ax.set_title(
        "Geographic Distribution of 671 CAMELS Basins: Mean Annual Precipitation",
        fontsize=12,
        fontweight="bold",
        pad=10,
        color="#0f172a",
    )

    ax.set_aspect("equal")
    ax.axis("off")
    bounds = conus.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.02
    pad_y = (bounds[3] - bounds[1]) * 0.03
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    out_file = OUTPUT_DIR / "fig02_camels_conus_precipitation.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_file}")

    if ROOT_OTHER_DIR.exists():
        fig.savefig(ROOT_OTHER_DIR / "fig02_camels_conus_precipitation.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_hydroclimatic_4panel(conus: gpd.GeoDataFrame, basins: gpd.GeoDataFrame):
    """Plot comprehensive 4-panel hydroclimatic characterization across CONUS."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300, constrained_layout=True)
    conus_boundary = conus.dissolve()

    panels = [
        {
            "ax": axes[0, 0],
            "title": "(a) 671 CAMELS Streamflow Gauges (CONUS)",
            "color_var": None,
            "color": "#2563eb",
            "label": "Gauged Catchments (N = 671)",
        },
        {
            "ax": axes[0, 1],
            "title": "(b) Mean Annual Precipitation (mm yr⁻¹)",
            "color_var": "p_annual_mm",
            "cmap": "YlGnBu",
            "vmin": 200,
            "vmax": 2200,
            "cbar_label": "Precipitation (mm yr⁻¹)",
        },
        {
            "ax": axes[1, 0],
            "title": "(c) Aridity Index (PET / P)",
            "color_var": "aridity",
            "cmap": "Spectral_r",
            "vmin": 0.3,
            "vmax": 2.2,
            "cbar_label": "Aridity Index (PET / P)",
        },
        {
            "ax": axes[1, 1],
            "title": "(d) Fraction of Precipitation Falling as Snow",
            "color_var": "frac_snow",
            "cmap": "Blues",
            "vmin": 0.0,
            "vmax": 0.8,
            "cbar_label": "Snow Fraction",
        },
    ]

    bounds = conus.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.02
    pad_y = (bounds[3] - bounds[1]) * 0.03

    for p in panels:
        ax = p["ax"]
        conus_boundary.plot(ax=ax, color="#f8fafc", edgecolor="none", zorder=1)
        conus.plot(ax=ax, color="none", edgecolor="#cbd5e1", linewidth=0.5, zorder=2)
        conus_boundary.plot(ax=ax, color="none", edgecolor="#475569", linewidth=0.9, zorder=3)

        if p["color_var"] is None or p["color_var"] not in basins.columns:
            ax.scatter(
                basins.geometry.x,
                basins.geometry.y,
                color=p["color"],
                s=16,
                alpha=0.85,
                edgecolors="#0f172a",
                linewidths=0.35,
                zorder=4,
            )
            ax.legend(
                [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=p['color'], markeredgecolor='#0f172a', markersize=6)],
                [p["label"]],
                loc="lower left",
                frameon=True,
                framealpha=0.9,
                fontsize=8,
            )
        else:
            sc = ax.scatter(
                basins.geometry.x,
                basins.geometry.y,
                c=basins[p["color_var"]],
                cmap=p["cmap"],
                vmin=p["vmin"],
                vmax=p["vmax"],
                s=16,
                alpha=0.88,
                edgecolors="#0f172a",
                linewidths=0.35,
                zorder=4,
            )
            cax = inset_axes(
                ax,
                width="34%",
                height="4.5%",
                loc="lower left",
                bbox_to_anchor=(0.04, 0.06, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
            cbar.set_label(p["cbar_label"], fontsize=8, labelpad=3)
            cbar.ax.tick_params(labelsize=7)

        ax.set_title(p["title"], fontsize=10.5, fontweight="bold", pad=2, color="#0f172a")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    out_file = OUTPUT_DIR / "fig01_camels_conus_hydroclimatic_4panel.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_file}")

    if ROOT_OTHER_DIR.exists():
        fig.savefig(ROOT_OTHER_DIR / "fig01_camels_conus_hydroclimatic_4panel.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def main():
    print("=== Generating CAMELS CONUS Geographic Distribution Maps ===")
    conus = ensure_us_states_geojson()
    basins = load_merged_data()
    print(f"Successfully loaded {len(basins)} basins across {len(conus)} CONUS states.")

    print("\n1. Rendering primary distribution map (fig02_camels_conus_distribution.png)...")
    plot_primary_conus_distribution(conus, basins)

    print("2. Rendering precipitation gradient map (fig02_camels_conus_precipitation.png)...")
    plot_precipitation_distribution(conus, basins)

    print("3. Rendering 4-panel hydroclimatic characterization (fig01_camels_conus_hydroclimatic_4panel.png)...")
    plot_hydroclimatic_4panel(conus, basins)

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
