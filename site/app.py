# app.py
import streamlit as st
import geopandas as gpd
from shapely.geometry import mapping
from supabase import create_client, Client
from solar_suitability import SolarSuitability

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



# MAIN
analyze_button = st.sidebar.button("Analyze", key="sidebar_analyze")
with tab1: 
    st.markdown(" A tool to reveal ideal rooftop placement of photovoltaic panels to meet predetermined urban energy needs.")  # this is a sub-heading

    # Button
    if analyze_button:
      
        planner = SolarSuitability(
        city=city,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        solar_pct=solar_pct,
        commercial_pct=commercial_pct,
        insolation_override=insolation_override
        )

        # Demand
        with st.spinner("Loading demand data from Supabase..."):
            demand_df = planner.load_demand_table()
    
        if df_demand.empty:
            st.error("Loaded demand table is empty. Check Supabase table names and permissions.")
            st.stop()
    
        annual_kwh = planner.compute_city_annual_kwh(demand_df)
        target_kwh = annual_kwh * (solar_pct / 100.0)

        if pd.isna(annual_kwh) or annual_kwh == 0:
            st.error(f"Annual kWh is {annual_kwh}. Cannot proceed.")
            st.stop()

        required_kwh = annual_kwh * (solar_pct / 100.0)
      
        # Load building footprints
        with st.spinner("Loading OSM buildings…"):
           gdf = planner.fetch_buildings_osm()

        if gdf.empty:
            st.error("Loaded demand table is empty. Check Supabase table names and permissions.")
            st.stop()

        # Estimate roof geometry / tilt
        gdf = planner.estimate_tilt_from_height_and_span(gdf)

        # Shade estimation (if you have GeoTIFF, else assume zero)
        if "shade" not in gdf.columns:
            gdf["shade"] = 0  # default — override later if needed

        # Suitability Scoring
        ideal_azimuth = demand_df["azimuth_sunrise"].iloc[-1] if "azimuth_sunrise" in demand_df else 150

        gdf = planner.compute_suitability_scores(
            gdf,
            irr_factor=1.0,
            ideal_azimuth=ideal_azimuth
        )

        # Annual Solar Potential
        gdf = planner.estimate_building_annual_potential(gdf)
        
        # Select buildings to meet target
        selected_gdf, total_kwh, count, actual_commercial_pct = planner.select_buildings(
            gdf,
            required_kwh=target_kwh,
            commercial_pct=commercial_pct
        )

        # Split into residential and commercial
        gdf_residential = selected_gdf[selected_gdf['is_residential']].copy()
        gdf_commercial = selected_gdf[~selected_gdf['is_residential']].copy()
        
        # Ensure they're still GeoDataFrames
        if not isinstance(gdf_residential, gpd.GeoDataFrame):
            gdf_residential = gpd.GeoDataFrame(gdf_residential, geometry='geometry', crs=selected_gdf.crs)
        if not isinstance(gdf_commercial, gpd.GeoDataFrame):
            gdf_commercial = gpd.GeoDataFrame(gdf_commercial, geometry='geometry', crs=selected_gdf.crs)

        # Plot results
        st.header("Map of Suitable Locations")
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
        st.header("City Specifications")
        specs_container = st.container(border=True)
        with specs_container:
            total_rooftop_area = gdf['area_m2'].sum()
            total_usable_area = gdf['usable_area_m2'].sum()
            total_selected_area = selected_gdf['area_m2'].sum()
            total_selected_usable_area = selected_gdf['usable_area_m2'].sum()

            st.write(f"Residential buildings: {gdf['is_residential'].sum()}")
            st.write(f"Commercial buildings: {(~gdf['is_residential']).sum()}")
            st.write(f"Total rooftop area (all buildings): {total_rooftop_area:,.0f} m²")
            st.write(f"Total usable area (all buildings): {total_usable_area:,.0f} m²")
            st.write("City annual energy (kWh)", f"{annual_kwh:,.0f}")
            #st.write("Current cost:")

        # Results container
        st.header("Results")
        specs_container = st.container(border=True)
        with specs_container:
            st.write(f"Selected {count} buildings")
            st.write(f"Total kWh selected: {tot_kwh:,.0f}")
            st.write(f"Actual commercial %: {actual_commercial_pct:.1f}%")
            st.write(f"Total potential all buildings: {gdf['annual_potential_kwh'].sum():,.0f} kWh")
            #st.write("Number of residential buildings used:")
            #st.write("Estimated number of panels required: " + '' * 25 + "Estimated total panel cost: ")
            #st.write("Payback period:")
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



