from __future__ import annotations

from pathlib import Path

import streamlit as st
from supabase import create_client, Client
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import tempfile
import requests
from datetime import datetime, timedelta

import rasterio
from rasterio.mask import mask
import osmnx as ox
import shapely
from shapely.geometry import Polygon, mapping
import pydeck as pdk
import json

from IPython.display import Image
import matplotlib.pyplot as pyplot
import folium
import folium.features
from streamlit_folium import st_folium
from folium.plugins import GroupedLayerControl

# Load secrets
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_ANON_KEY"]

supabase: Client = create_client(url, key)

st.set_page_config(layout="wide", page_title="Solar Suitability Planner")

# Default avg daily insolation
DEFAULT_INSOLATION = {
    "Ann Arbor": 4.0,   # ~kWh/m2/day 
    "Tucson": 6.0,
}
PANEL_EFFICIENCY = 0.18   # guessing, fill in from PV table
PERFORMANCE_RATIO = 0.75  # guessing, fill in from PV table

@st.cache_data(show_spinner=False)
def load_demand_table(city: str):
    """
    Load city demand table from Supabase.
    """
    table_name = "Ann_Arbor_demand" if city == "Ann Arbor" else "TEPC_demand"
    res = supabase.table(table_name).select("*").limit(100000).execute()
    data = res.data
    df = pd.DataFrame(data)

    # Convert data types - working on correcting OG csv and db so this can be removed
    if 'MW' in df.columns:
        df['MW'] = pd.to_numeric(df['MW'], errors='coerce')
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Parse azimuth columns if they exist (remove degree symbol)
    if 'Azimuth_Sunrise' in df.columns:
        df['azimuth_sunrise'] = df['Azimuth_Sunrise'].str.replace('°', '').astype(float)
    if 'Azimuth_Sunset' in df.columns:
        df['azimuth_sunset'] = df['Azimuth_Sunset'].str.replace('°', '').astype(float)
        
    return df

def fetch_buildings_osm(place_name: str):
    """
    Use OSMnx to fetch building footprints for the given place name.
    Returns GeoDataFrame with area_m2 computed per building and total area.
    """
    tags = {"building": ['apartments', 'residential', 'garage', 'detached', 'bungalow', 'house', 'semidetached_house']}
    r_gdf = ox.features.features_from_place(place_name, tags=tags)

    tags = {"building": ['office', 'university', 'yes',  'train_station', 'courthouse', 'hospital', 'industrial', 
                         'warehouse', 'hotel', 'commercial', 'roof', 'public', 'carport', 'parking', 'retail', 
                         'college', 'yes;no', 'storage_tank', 'central_office', 'terrace', 'garages', 'civic', 
                         'government', 'data_center']}
    c_gdf = ox.features.features_from_place(place_name, tags=tags)

    gdf = pd.concat([r_gdf, c_gdf])

    drop_labels = ['museum', 'stable', 'service', 'greenhouse', 'kindergarten', 'hangar', 'static_caravan', 
                      'collapsed', 'ses', 'bunker', 'kiosk', 'no', 'sports_centre', 'chapel', 'historic', 
                      'grandstand','dormitory', 'school', 'church', 'shed', 'cathedral', 'stadium', 'ruins', 
                      'pavilion', 'toilets', 'motel', 'container', 'construction', 'fire_station', 'farm_auxiliary']
    
    gdf = gdf[~gdf["building"].isin(drop_labels)]
    
    # Filter to only Polygons and MultiPolygons
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    
    # Residential/Commercial Classification
    gdf["is_residential"] = gdf.apply(
        lambda row: (
            (pd.notna(row.get("building")) and str(row.get("building")).lower() in
             ['apartments', 'residential', 'garage', 'detached', 'bungalow', 'house', 'semidetached_house'])
        ), axis=1
    )

    # Calculate the area (meters)
    gdf_proj = ox.projection.project_gdf(gdf)
    gdf['area_m2'] = gdf_proj.area

    # Total area (sq m) covered by building footprints - this will go to City Specs
    total_area = gdf['area_m2'].sum()

    # put in some logic regarding the CNN identification of rooftops with PV, 
    # which will need to be discarded

    return gdf   #leave off total_area for now

def compute_city_annual_kwh(df: pd.DataFrame):
    """
    Convert the 15-minute MW measurements into annual kWh.
    Each MW measurement over a 15-min interval contributes: MW * 0.25 hours = MWh.
    Sum of (MW) * 0.25 gives MWh per dataset period; multiply by 1000 to kWh.
    Returns predicted value of next year's energy demand
    """
    # For each year: sum(MW) * 0.25 hours = MWh -> convert to kWh
    summary = (
        df.groupby('Year')['MW']
        .sum()
        .reset_index(name='total_MW_sum')
    )
    summary['annual_kWh'] = summary['total_MW_sum'] * 0.25 * 1000

    # Sort by year and calculate growth rate
    summary = summary.sort_values("Year").reset_index(drop=True)
    summary["growth_rate"] = summary["annual_kWh"].pct_change()
    
    # Get most recent year's data
    last_year = summary.iloc[-1]["Year"]
    last_demand = summary.iloc[-1]["annual_kWh"]
    last_growth = summary.iloc[-1]["growth_rate"]
    
    st.write(f"- Last year: {int(last_year)}")
    st.write(f"- Last demand: {last_demand:,.0f} kWh")
    st.write(f"- Last growth rate: {last_growth}")
    
    # If only one year of data (growth_rate will be NaN), assume 2% growth
    if pd.isna(last_growth):
        last_growth = 0.02
        st.info(f"ℹ️ Only one year of data ({int(last_year)}). Assuming 2% annual growth.")
    
    # Predict next year's demand
    predicted = last_demand * (1 + last_growth)

    return predicted

def polygon_orientation(polygon: Polygon):
    """
    Estimate roof orientation (degrees clockwise from North) by minimum_rotated_rectangle method.
    Returns degrees 0-360 where 0 = East? We'll return as degrees clockwise from North:
    convert from arctan2(dx, dy) with adjustments.
    """
    if polygon.is_empty:
        return np.nan
    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    max_len = 0
    best_angle = None
    # iterate edges
    for i in range(len(coords)-1):
        x1,y1 = coords[i]
        x2,y2 = coords[i+1]
        dx = x2 - x1
        dy = y2 - y1
        edge_len = np.hypot(dx, dy)
        if edge_len > max_len:
            max_len = edge_len
            best_angle = np.degrees(np.arctan2(dy, dx))
    if best_angle is None:
        return np.nan
    # best_angle is angle from +x axis (East) CCW. Convert to degrees clockwise from North:
    # angle_from_north_clockwise = (90 - best_angle) mod 360
    angle = (90 - best_angle) % 360
    return angle

def estimate_tilt_from_height_and_span(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Estimate roof tilt in degrees using any available height / building:levels or fallback heuristics.

    - If 'height' available (meters), use it
    - Else if 'building:levels' available, multiply by 3.2
    - Else fallback to small heights with default tilt for commercial vs residential
    """
    # handle numeric heights
    if "height" in gdf.columns:
        heights = pd.to_numeric(gdf["height"], errors="coerce")
    elif "building:levels" in gdf.columns:
        heights = pd.to_numeric(gdf["building:levels"], errors="coerce") * 3.2
    else:
        heights = pd.Series(np.nan, index=gdf.index)

    # Effective roof span (approx): use sqrt(area) or MRR longest edge-derived width
    def approx_width(geom):
        try:
            mrr = geom.minimum_rotated_rectangle
            # approximate width by dividing perimeter by 4 (rectangle assumption)
            perim = mrr.length
            width = perim / 4.0
            return max(width, np.sqrt(geom.area))
        except Exception:
            return np.sqrt(geom.area)

    gdf["width_m"] = gdf.geometry.to_crs(epsg=3857).apply(lambda geom: approx_width(geom))
    heights_filled = heights.fillna(5.0)
    # tilt = arctan(height / (width/2)) [approx roof pitch]
    gdf["tilt_deg"] = np.degrees(np.arctan2(heights_filled, gdf["width_m"]/2.0))
    # clamp sensible range
    gdf["tilt_deg"] = gdf["tilt_deg"].clip(3, 45)
    return gdf

def orientation_match_score(roof_angle_deg: float, ideal_sun_azimuth_deg: float):
    """
    Compute orientation match score 0-1 where 1 = perfect face to sun,
    using smallest absolute difference on circle (0..180)
    """
    diff = abs(roof_angle_deg - ideal_sun_azimuth_deg) % 360
    if diff > 180:
        diff = 360 - diff
    # map 0->1, 180->0
    return 1.0 - (diff / 180.0)

def estimate_building_annual_potential_kwh(gdf: gpd.GeoDataFrame, insolation_kwh_m2_day: float,
                                           panel_efficiency=PANEL_EFFICIENCY, perf_ratio=PERFORMANCE_RATIO):
    """
    Estimate the annual kWh each building can produce.
    """
   # Set limit on useable area
    COMMERCIAL_FRACTION = 0.50
    RESIDENTIAL_FRACTION = 0.25 

    # Compute usable area
    gdf["usable_fraction"] = gdf["is_residential"].apply(
        lambda x: RESIDENTIAL_FRACTION if x else COMMERCIAL_FRACTION
    )
    gdf["usable_area_m2"] = gdf["area_m2"] * gdf["usable_fraction"]

    # Energy factor based on insolation + efficiency
    daily = float(insolation_kwh_m2_day)
    factor = daily * 365.0 * panel_efficiency * perf_ratio
    gdf["annual_potential_kwh"] = gdf["usable_area_m2"] * factor

    return gdf

def select_buildings_to_meet_target(gdf: gpd.GeoDataFrame, required_kwh: float, commercial_pct: float):
    """
    Greedy selection that alternates choosing commercial/residential in a ratio
    that tries to match the requested commercial_pct while picking highest solar_score first.
    Returns GeoDataFrame of selected buildings and remaining totals.
    """
    # split pools
    comm = gdf[gdf["is_residential"]==False].sort_values("solar_score", ascending=False).copy()
    resid = gdf[gdf["is_residential"]==True].sort_values("solar_score", ascending=False).copy()

    # iteratively pick from pools to reach required_kwh
    sel_rows = []
    ptr_comm = 0
    ptr_resid = 0
    total_selected_kwh = 0.0
    selected_comm = 0
    selected_total = 0

    # Avoid division by zero: desired fraction [0..1]
    desired_comm_frac = commercial_pct / 100.0

    # Safety: if both pools empty, return empty
    while total_selected_kwh < required_kwh and (ptr_comm < len(comm) or ptr_resid < len(resid)):
        # choose which pool to pick from based on current fraction
        current_frac = (selected_comm / selected_total) if selected_total > 0 else 0.0
        pick_comm = False
        # If we have not yet reached desired commercial fraction and comm pool not empty -> pick commercial
        if current_frac < desired_comm_frac and ptr_comm < len(comm):
            pick_comm = True
        # else pick residential if available
        elif ptr_resid < len(resid):
            pick_comm = False
        # else fallback to commercial if residential exhausted
        elif ptr_comm < len(comm):
            pick_comm = True
        else:
            break

        if pick_comm:
            row = comm.iloc[ptr_comm]
            ptr_comm += 1
        else:
            row = resid.iloc[ptr_resid]
            ptr_resid += 1

        sel_rows.append(row)
        total_selected_kwh += float(row["annual_potential_kwh"])
        selected_total += 1
        if not row["is_residential"]:
            selected_comm += 1

    selected_gdf = gpd.GeoDataFrame(sel_rows, crs=gdf.crs).reset_index(drop=True) if len(sel_rows) > 0 else gpd.GeoDataFrame(columns=gdf.columns)
    # compute metrics
    actual_comm_pct = (selected_comm / selected_total * 100.0) if selected_total > 0 else 0.0
    return selected_gdf, total_selected_kwh, selected_total, actual_comm_pct

def compute_suitability_scores(gdf: gpd.GeoDataFrame, irradiance_factor: float, ideal_sun_azimuth: float):
    """
    Compose final solar_score from orientation, tilt, shade, irradiance and area.
    We compute intermediate scores in [0,1] and combine with weights.
    """
    # orientation
    gdf["orientation_deg"] = gdf.geometry.apply(lambda geom: polygon_orientation(geom))
    gdf["orientation_score"] = gdf["orientation_deg"].apply(lambda a: orientation_match_score(a, ideal_sun_azimuth))
    # tilt: ideal ~25 deg (residential) -> score drops as further away
    gdf["tilt_score"] = 1.0 - (np.abs(gdf["tilt_deg"] - 25.0) / 40.0)
    gdf["tilt_score"] = gdf["tilt_score"].clip(0,1)
    # shade already 0..1 where 1 is shaded; we want exposure_score = 1 - shade
    gdf["exposure_score"] = (1.0 - gdf["shade"]).clip(0,1)
    # irradiance_factor applied uniformly (0..1)
    irr = float(np.clip(irradiance_factor, 0.0, 1.0))
    # combine: weights can be tuned
    gdf["solar_score"] = (
        0.35 * gdf["orientation_score"] +
        0.25 * gdf["tilt_score"] +
        0.25 * gdf["exposure_score"] +
        0.15 * irr
    )
    gdf["solar_score"] = gdf["solar_score"].clip(0,1)
    return gdf

# STREAMLIT
#------------------------------
st.title("Urban Solar Suitability Planner")
st.markdown(" A tool to reveal ideal rooftop placement of photovoltaic panels to meet predetermined urban energy needs.")  # this is a sub-heading

# Sidebar controls
city = st.sidebar.selectbox("City", ["Ann Arbor", "Tucson"])
solar_pct = st.sidebar.slider("Percent of city energy to meet with solar", 1, 100, 30)
commercial_pct = st.sidebar.slider("Percent of selected buildings to be commercial", 0, 100, 20)
insolation_override = st.sidebar.number_input("Insolation (kWh/m²/day) — optional override",        # this is a +/- counter
                                              value=float(DEFAULT_INSOLATION[city]), min_value=0.0, step=0.1)

# Button
if st.sidebar.button("Analyze"):
    
    # Load demand data
    with st.spinner("Loading demand data from Supabase..."):
        df_demand = load_demand_table(city)
    
    if df_demand.empty:
        st.error("Loaded demand table is empty. Check Supabase table names and permissions.")
        st.stop()

    # Compute annual energy
    st.write("**Step 2: Computing annual energy need...**")
    with st.spinner("Computing annual energy need..."):
        annual_kwh = compute_city_annual_kwh(df_demand)
    
    st.write(f"Annual kWh result: {annual_kwh}")
    
    if pd.isna(annual_kwh) or annual_kwh == 0:
        st.error(f"Annual kWh is {annual_kwh}. Cannot proceed.")
        st.stop()

    required_kwh = annual_kwh * (solar_pct / 100.0)

    # Rasters
    tif_name = "ann_arbor.tif" if city == "Ann Arbor" else "tucson.tif"
    raster = None
    try:
        with st.spinner("Downloading GeoTIFF from Supabase..."):
            # Check if download_geotiff_from_supabase function works
            if 'download_geotiff_from_supabase' in globals():
                raster = download_geotiff_from_supabase("satellite", tif_name)
                st.write("GeoTIFF loaded")
            else:
                st.warning("Download_geotiff_from_supabase function not found. Skipping raster.")
    except Exception as e:
        st.warning(f"Could not download GeoTIFF: {e}. Continuing without shade (shade=0).")

    # Get buildings
    with st.spinner("Fetching building footprints from OSM (this may take a minute)..."):
        place_name = f"{city}, USA"
        try:
            buildings = fetch_buildings_osm(place_name)
            st.write(f"✓ Fetched {len(buildings)} buildings")
            st.write(f"Columns: {buildings.columns.tolist()}")
        except Exception as e:
            st.error(f"OSMnx fetch failed: {e}")
            st.stop()

    if len(buildings) == 0:
        st.error("No buildings found")
        st.stop()

    # Compute geometry features
    with st.spinner("Estimating roof geometry (orientation, tilt, area)..."):
        buildings["orientation_deg"] = buildings.geometry.apply(lambda g: polygon_orientation(g))
        buildings = estimate_tilt_from_height_and_span(buildings)

    # Compute shade
    if raster is not None:
        st.info("Computing shade from GeoTIFF...")
        # Check if shade_from_geotiff function exists
        if 'shade_from_geotiff' in globals():
            progress = st.progress(0)
            total = len(buildings)
            shades = []
            for i, (idx, row) in enumerate(buildings.iterrows()):
                try:
                    s = shade_from_geotiff(raster, row.geometry)
                except Exception:
                    s = 0.0
                shades.append(s)
                if i % 100 == 0 and total > 0:
                    progress.progress(int((i+1)/total * 100))
            buildings["shade"] = shades
            progress.empty()
            st.write(f"Shade range: {buildings['shade'].min():.2f} to {buildings['shade'].max():.2f}")
        else:
            st.warning("Shade_from_geotiff function not found. Setting shade=0")
            buildings["shade"] = 0.0
    else:
        buildings["shade"] = 0.0
        st.write("✓ Shade set to 0.0 (no raster)")

    # Irradiance and azimuth
    st.write("**Step 7: Computing irradiance factor and ideal azimuth...**")
    irr_factor = np.clip(insolation_override / 7.0, 0.0, 1.0)
    
    if {"azimuth_sunrise","azimuth_sunset"}.issubset(df_demand.columns):
        try:
            mean_azimuth_range = ((df_demand["azimuth_sunset"] + df_demand["azimuth_sunrise"]) / 2.0).mean()
            ideal_azimuth = float(mean_azimuth_range)
            st.write(f"✓ Ideal azimuth: {ideal_azimuth:.1f}° (from data)")
        except Exception:
            ideal_azimuth = 180.0
            st.write(f"✓ Ideal azimuth: 180° (default)")
    else:
        ideal_azimuth = 180.0
        st.write(f"✓ Ideal azimuth: 180° (default)")

    # Compute suitability scores
    with st.spinner("Computing solar suitability scores and annual potential..."):
        buildings = compute_suitability_scores(buildings, irr_factor, ideal_azimuth)
        buildings = estimate_building_annual_potential_kwh(buildings, insolation_override)
    
    with st.spinner("Selecting buildings to meet target while respecting commercial mix..."):
        selected_gdf, tot_kwh_sel, n_selected, actual_comm_pct = select_buildings_to_meet_target(buildings, required_kwh, commercial_pct)

    # Split into residential and commercial
    gdf_residential = selected_gdf[selected_gdf['is_residential']].copy()
    gdf_commercial = selected_gdf[~selected_gdf['is_residential']].copy()
    
    # Ensure they're still GeoDataFrames
    if not isinstance(gdf_residential, gpd.GeoDataFrame):
        gdf_residential = gpd.GeoDataFrame(gdf_residential, geometry='geometry', crs=selected_gdf.crs)
    if not isinstance(gdf_commercial, gpd.GeoDataFrame):
        gdf_commercial = gpd.GeoDataFrame(gdf_commercial, geometry='geometry', crs=selected_gdf.crs)

    # Plot results
    map_container = st.container(border=True)
    with map_container:
        if not gdf_residential.empty and not gdf_commercial.empty:
            fig, ax = ox.plot.plot_footprints(
                gdf_residential, 
                figsize=(10, 8), 
                color='blue', 
                ax=None,
                save=False,
                show=False,
                close=False
            )
            ox.plot.plot_footprints(
                gdf_commercial, 
                color='red', 
                ax=ax,
                save=False,
                show=False,
                close=False
            )
        elif not gdf_residential.empty:
            fig, ax = ox.plot.plot_footprints(
                gdf_residential, 
                figsize=(10, 8), 
                color='blue', 
                save=False,
                show=False,
                close=False
            )
        elif not gdf_commercial.empty:
            fig, ax = ox.plot.plot_footprints(
                gdf_commercial, 
                figsize=(10, 8), 
                color='red', 
                save=False,
                show=False,
                close=False
            )
        else:
            st.warning("No buildings to display.")
            fig = None

    if fig is not None:
        st.pyplot(fig)
        
    # City specs container
    st.write("City Specs")
    with st.container(height=300):
        st.write("Number of residential buildings:" + '' * 25 + "Available m2:")
        st.write("Number of commercial buildings:" + '' * 25 + "Available m2:")
        st.metric("City annual energy (kWh)", f"{annual_kwh:,.0f}")
        st.write("Projected power demand: " + '' * 25 + "Growth rate: ")
        st.write("Current cost:")

    # Results container
    st.write("Results")
    with st.container(height=300):
        st.write(f"Actual commercial %: {actual_comm_pct:.1f}%")
        st.write("Number of residential buildings used:")
        st.write("Estimated number of panels required: " + '' * 25 + "Estimated total panel cost: ")
        st.write("Payback period:")
        st.write(f"✓ Total potential all buildings: {buildings['annual_potential_kwh'].sum():,.0f} kWh")
        st.write(f"✓ Residential buildings: {buildings['is_residential'].sum()}")
        st.write(f"✓ Commercial buildings: {(~buildings['is_residential']).sum()}")
        #st.write("First year savings: " + "25 year savings: ")
        #st.write("Annual CO2 redux: " + "25 year CO2 redux")
    
    st.success("Analysis complete")
