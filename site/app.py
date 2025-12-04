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
def load_demand_table(city: str) -> pd.DataFrame:
    """
    Load city demand table from Supabase.
    """
    table_name = "Ann_Arbor_demand" if city == "Ann Arbor" else "TEPC_demand"
    res = supabase.table(table_name).select("*").execute()
    data = res.data
    df = pd.DataFrame(data)
    return df

st.title("Urban Solar Suitability Planner")
st.markdown(" A tool to reveal ideal rooftop placement of photovoltaic panels to meet predetermined urban energy needs.")  # this is a sub-heading

# Sidebar controls
city = st.sidebar.selectbox("City", ["Ann Arbor", "Tucson"])
solar_pct = st.sidebar.slider("Percent of city energy to meet with solar", 1, 100, 30)
commercial_pct = st.sidebar.slider("Percent of selected buildings to be commercial", 0, 100, 20)
insolation_override = st.sidebar.number_input("Insolation (kWh/m²/day) — optional override",        # this is a +/- counter
                                              value=float(DEFAULT_INSOLATION[city]), min_value=0.0, step=0.1)

# Map container
map_container = st.container(border=True)
with map_container:
    # Layered version with R/C classification
    # Get residential buildings
    r_tags = {"building": ['apartments', 'residential', 'garage', 'detached', 'bungalow', 'house', 'semidetached_house']}
    gdf_residential = ox.features.features_from_place(city, r_tags)
    
    # Get commercial/industrial buildings
    c_tags = {"building": ['office', 'university', 'yes',  'train_station', 'courthouse', 
                           'hospital', 'industrial', 'warehouse', 'hotel', 'commercial', 
                           'roof', 'public', 'carport', 'parking', 'retail', 'college', 'yes;no', 
                           'storage_tank', 'central_office', 'terrace', 'garages', 'civic', 'government', 'data_center']}
    gdf_commercial = ox.features.features_from_place(city, c_tags)
    
    # Create figure and plot both layers
    fig, ax = ox.plot.plot_footprints(
        gdf_residential, 
        figsize=(10, 8), 
        color='blue', 
        ax=None,  # Create new axes
        save=False,
        show=False,
        close=False
    )
    
    # Add commercial buildings to the same axes
    ox.plot.plot_footprints(
        gdf_commercial, 
        color='red', 
        ax=ax,  # Use the same axes
        save=False,
        show=False,
        close=False
    )
    
    # Add a legend - too large and ugly, change to inactive buttons?
    #from matplotlib.patches import Patch
    #legend_elements = [
    #    Patch(facecolor='blue', label='Residential'),
    #    Patch(facecolor='red', label='Commercial/Industrial')
    #]
    #ax.legend(handles=legend_elements, loc='lower left')
    
    st.pyplot(fig)

# City specs container
st.write("City Specs")
with st.container(height=300):
    st.write("Number of residential buildings:" + '' * 25 + "Available m2:")
    st.write("Number of commercial buildings:" + '' * 25 + "Available m2:")
    st.write("Average yearly power demand:")
    st.write("Projected power demand: " + '' * 25 + "Growth rate: ")
    st.write("Current cost:")

# Results container
st.write("Results")
with st.container(height=300):
    st.write("Number of commercial buildings used:")
    st.write("Number of residential buildings used:")
    st.write("Estimated number of panels required: " + '' * 25 + "Estimated total panel cost: ")
    st.write("Payback period:")
    #st.write("First year savings: " + "25 year savings: ")
    #st.write("Annual CO2 redux: " + "25 year CO2 redux")
