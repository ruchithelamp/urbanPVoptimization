# app.py
import streamlit as st
import geopandas as gpd
from shapely.geometry import mapping
import SolarSuitability

from utils.inference import roofarea
from utils.conversions import CITY_RULES
from utils.inference import roofarea


# Load secrets
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_ANON_KEY"]

supabase: Client = create_client(url, key)

st.set_page_config(layout="wide", page_title="Solar Suitability Planner")

st.markdown("A tool to reveal ideal rooftop placement of photovoltaic panels to meet predetermined urban energy needs.")

# Sidebar controls
city = st.sidebar.selectbox("City", ["Ann Arbor", "Tucson"])
solar_pct = st.sidebar.slider("Percent of city energy to meet with solar", 1, 100, 30)
commercial_pct = st.sidebar.slider("Percent of selected buildings to be commercial", 0, 100, 20)
insolation_override = st.sidebar.number_input("Insolation (kWh/m²/day) — optional override",
                                              value=float(DEFAULT_INSOLATION[city]), min_value=0.0, step=0.1)



# Sq ft TAB
analyze_button = st.sidebar.button("Analyze", key="sidebar_analyze")
with tab1: 
    st.markdown(" A tool to reveal ideal rooftop placement of photovoltaic panels to meet predetermined urban energy needs.")  # this is a sub-heading

    # Button
    if analyze_button:
        
        # Load demand data
        with st.spinner("Loading demand data from Supabase..."):
            df_demand = load_demand_table(city)
        
        if df_demand.empty:
            st.error("Loaded demand table is empty. Check Supabase table names and permissions.")
            st.stop()

        # Compute annual energy
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
                st.warning("Shade from geotiff not found. Setting shade=0")
                buildings["shade"] = 0.0
        else:
            buildings["shade"] = 0.0
            st.write("Shade set to 0.0 (no raster)")

        # Irradiance and azimuth
        irr_factor = np.clip(insolation_override / 7.0, 0.0, 1.0)
        
        if {"azimuth_sunrise","azimuth_sunset"}.issubset(df_demand.columns):
            try:
                mean_azimuth_range = ((df_demand["azimuth_sunset"] + df_demand["azimuth_sunrise"]) / 2.0).mean()
                ideal_azimuth = float(mean_azimuth_range)
                st.write(f"✓ Ideal azimuth: {ideal_azimuth:.1f}° (from data)")
            except Exception:
                ideal_azimuth = 180.0
        else:
            ideal_azimuth = 180.0

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
        

with tab2:
    st.header("Roof Area Estimator")

    st.write("Upload a 256x256 satellite tile, and choose the respective city. ")

    city_choose = st.selectbox(
        "Select your city", list(CITY_RULES.keys()), key="roof_city"
    )

    # image upload
    tile = st.file_uploader(
        "Upload satellite tile image (PNG)", type=["png", "jpg", "jpeg"]
    )

    if tile:
        st.image(tile, caption="Satellite Tile", width="stretch")

        if st.button("Calculate Roof Area"): 
            with st.spinner("Predicting..."):
                try:
                    area, mask = roofarea(tile, city_choose, supabase)
                    st.success(f"Roof Area Estimation: {area} sq ft")

                    st.subheader("Roofs found: ")
                    st.image(mask, caption="roof overlay", width="stretch")
                except Exception as e: 
                    st.error(f"Error: {e}")

    # thing



