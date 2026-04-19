# app.py
# MULTI-SUPABASE VERSION
#
# Demand tables stored in Supabase Account A
# Polygon tables stored in Supabase Account B
#
# Keeps:
# - demand forecasting
# - polygon area calculations
# - solar suitability logic
#
# ==============================================================
# pip install streamlit pandas geopandas shapely plotly supabase
# ==============================================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape
from supabase import create_client
import plotly.express as px

# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    layout="wide",
    page_title="Urban Solar Suitability Planner"
)

# ==============================================================
# TWO SUPABASE CONNECTIONS
# ==============================================================

# --------------------------
# DEMAND DATABASE
# --------------------------
supabase_demand = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_ANON_KEY"]
)

# --------------------------
# BUILDING DATABASE
# --------------------------
supabase_buildings = create_client(
    st.secrets["SUPABASE_BUILDING_URL"],
    st.secrets["SUPABASE_BUILDING_KEY"]
)

# ==============================================================
# TABLE MAPS
# ==============================================================

CITY_TABLE_MAP = {
    "Ann Arbor":"Ann_Arbor_demand",
    "Homestead":"Homestead_demand",
    "Los Angeles":"LA_demand",
    "Portland":"Portland_demand",
    "Seattle":"Seattle_demand",
    "Tacoma":"Tacoma_demand",
    "Tallahassee":"Tallahassee_demand",
    "Tampa":"Tampa_demand",
    "Tucson":"TEPC_demand"
}

CITY_BUILDING_MAP = {
    "Ann Arbor":"ann_arbor_buildings_geojson",
    "Homestead":"homestead_buildings_geojson",
    "Los Angeles":"los_angeles_buildings_geojson",
    "Portland":"portland_buildings_geojson",
    "Seattle":"seattle_buildings_geojson",
    "Tacoma":"tacoma_buildings_geojson",
    "Tallahassee":"tallahassee_buildings_geojson",
    "Tampa":"tampa_buildings_geojson",
    "Tucson":"tucson_buildings_geojson"
}

DEFAULT_INSOLATION = {
    "Ann Arbor":4.0,
    "Homestead":4.4,
    "Los Angeles":4.6,
    "Portland":3.0,
    "Seattle":2.9,
    "Tacoma":3.0,
    "Tallahassee":3.96,
    "Tampa":4.0,
    "Tucson":6.0
}

# ==============================================================
# LOAD DEMAND DATA
# ==============================================================

@st.cache_data(ttl=3600)
def load_yearly(city):

    table = CITY_TABLE_MAP[city]

    res = (
        supabase_demand
        .table(table)
        .select("Year,MW")
        .limit(1000000)
        .execute()
    )

    df = pd.DataFrame(res.data)

    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")

    out = (
        df.groupby("Year", as_index=False)["MW"]
        .sum()
    )

    out["annual_kwh"] = out["MW"] * 250

    return out.sort_values("Year")

# ==============================================================
# LOAD BUILDINGS FROM SECOND DATABASE
# ==============================================================

@st.cache_data(ttl=3600)
def load_buildings(city):

    table = CITY_BUILDING_MAP[city]

    res = (
        supabase_buildings
        .table(table)
        .select("*")
        .limit(50000)
        .execute()
    )

    df = pd.DataFrame(res.data)

    df["geometry"] = df["geometry"].apply(shape)

    gdf = gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs="EPSG:4326"
    )

    # Calculate area from polygons
    gdf = gdf.to_crs(3857)
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf.to_crs(4326)

    # If not already stored
    if "is_residential" not in gdf.columns:

        res_types = [
            "apartments",
            "residential",
            "detached",
            "bungalow",
            "house",
            "semidetached_house"
        ]

        gdf["is_residential"] = gdf["building"].isin(res_types)

    if "address" not in gdf.columns:
        gdf["address"] = gdf.get("name", "")

    return gdf

# ==============================================================
# DEMAND FORECAST
# ==============================================================

def forecast(df):

    vals = df.tail(3)["annual_kwh"].to_numpy()

    w = np.array([0.2,0.3,0.5])[-len(vals):]
    w = w / w.sum()

    return float((vals * w).sum())

# ==============================================================
# SOLAR CALCULATIONS
# ==============================================================

def run_selection(gdf, required, insolation):

    gdf = gdf.copy()

    gdf["usable_area_m2"] = np.where(
        gdf["is_residential"],
        gdf["area_m2"] * 0.25,
        gdf["area_m2"] * 0.50
    )

    gdf["annual_potential_kwh"] = (
        gdf["usable_area_m2"]
        * insolation
        * 365
        * 0.18
        * 0.75
    )

    gdf["solar_score"] = (
        gdf["annual_potential_kwh"]
        / gdf["annual_potential_kwh"].max()
    )

    gdf = gdf.sort_values(
        "solar_score",
        ascending=False
    )

    gdf["cum_kwh"] = gdf["annual_potential_kwh"].cumsum()

    return gdf[gdf["cum_kwh"] <= required]

# ==============================================================
# MAP
# ==============================================================

def draw_map(gdf):

    if gdf.empty:
        st.info("No selected buildings.")
        return

    sample = gdf.head(1500).copy()

    sample["lat"] = sample.geometry.centroid.y
    sample["lon"] = sample.geometry.centroid.x

    fig = px.scatter_mapbox(
        sample,
        lat="lat",
        lon="lon",
        hover_name="address",
        hover_data={
            "annual_potential_kwh":":,.0f",
            "solar_score":":.2f"
        },
        zoom=11,
        height=700
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0,r=0,t=0,b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# UI
# ==============================================================

st.title("Urban Solar Suitability Planner")

city = st.sidebar.selectbox(
    "City",
    list(CITY_TABLE_MAP.keys())
)

solar_pct = st.sidebar.slider(
    "Percent of city demand met by solar",
    1,100,30
)

insolation = st.sidebar.number_input(
    "Daily Insolation",
    value=float(DEFAULT_INSOLATION[city]),
    step=0.1
)

# ==============================================================
# ANALYZE
# ==============================================================

if st.sidebar.button("Analyze"):

    with st.spinner("Loading demand data..."):
        yearly = load_yearly(city)

    demand = forecast(yearly)

    required = demand * solar_pct / 100

    with st.spinner("Loading building polygons..."):
        bld = load_buildings(city)

    with st.spinner("Running solar suitability model..."):
        sel = run_selection(
            bld,
            required,
            insolation
        )

    st.session_state["yearly"] = yearly
    st.session_state["demand"] = demand
    st.session_state["required"] = required
    st.session_state["sel"] = sel

# ==============================================================
# TABS
# ==============================================================

A,B,C,D = st.tabs(
    ["Summary","Map","Buildings","Export"]
)

with A:

    if "demand" in st.session_state:

        st.metric(
            "Forecast Demand (kWh)",
            f"{st.session_state['demand']:,.0f}"
        )

        st.metric(
            "Solar Target (kWh)",
            f"{st.session_state['required']:,.0f}"
        )

        st.line_chart(
            st.session_state["yearly"]
            .set_index("Year")["annual_kwh"]
        )

with B:

    draw_map(
        st.session_state.get(
            "sel",
            gpd.GeoDataFrame()
        )
    )

with C:

    sel = st.session_state.get(
        "sel",
        gpd.GeoDataFrame()
    )

    if not sel.empty:

        st.dataframe(
            sel[
                [
                    "address",
                    "is_residential",
                    "area_m2",
                    "annual_potential_kwh",
                    "solar_score"
                ]
            ]
        )

with D:

    sel = st.session_state.get(
        "sel",
        gpd.GeoDataFrame()
    )

    if not sel.empty:

        csv = (
            sel.drop(columns="geometry")
            .to_csv(index=False)
            .encode()
        )

        st.download_button(
            "Download CSV",
            csv,
            file_name="solar_results.csv"
        )
