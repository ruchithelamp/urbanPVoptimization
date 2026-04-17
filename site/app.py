# app.py
import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
import osmnx as ox
import geopandas as gpd
import folium
from streamlit_folium import st_folium

st.set_page_config(layout='wide', page_title='Urban Solar Suitability Planner')

# ---------------- Config ----------------
CITY_TABLE_MAP = {
    'Ann Arbor':'Ann_Arbor_demand','Homestead':'Homestead_demand','Los Angeles':'LA_demand',
    'Portland':'Portland_demand','Seattle':'Seattle_demand','Tacoma':'Tacoma_demand',
    'Tallahassee':'Tallahassee_demand','Tampa':'Tampa_demand','Tucson':'TEPC_demand'
}
DEFAULT_INSOLATION = {'Ann Arbor':4.0,'Homestead':4.4,'Los Angeles':4.6,'Portland':3.0,'Seattle':2.9,'Tacoma':3.0,'Tallahassee':3.96,'Tampa':4.0,'Tucson':6.0}

supabase = create_client(st.secrets['SUPABASE_URL'], st.secrets['SUPABASE_ANON_KEY'])

# ---------------- Data Layer ----------------
@st.cache_data(ttl=3600)
def load_yearly_demand(city):
    table = CITY_TABLE_MAP[city]
    res = supabase.table(table).select('Year,MW').limit(1000000).execute()
    df = pd.DataFrame(res.data)
    df['MW'] = pd.to_numeric(df['MW'], errors='coerce')
    out = df.groupby('Year', as_index=False)['MW'].sum()
    out['annual_kwh'] = out['MW'] * 250
    return out.sort_values('Year')

@st.cache_data(ttl=86400)
def load_buildings(city):

    res_tags = {"building": [
        "house","detached","residential",
        "apartments","semidetached_house"
    ]}

    com_tags = {"building": [
        "office","commercial","retail",
        "warehouse","industrial","hotel"
    ]}

    r = ox.features_from_place(f"{city}, USA", tags=res_tags)
    c = ox.features_from_place(f"{city}, USA", tags=com_tags)

    gdf = pd.concat([r, c])

    gdf["is_residential"] = gdf["building"].isin(
        ["house","detached","residential",
         "apartments","semidetached_house"]
    )

    gdf = gdf[gdf.geometry.type.isin(["Polygon","MultiPolygon"])]

    return gdf

# ---------------- Analytics ----------------
def forecast_next_year(df):
    recent = df.tail(3)
    vals = recent['annual_kwh'].to_numpy()
    w = np.array([0.2,0.3,0.5])[-len(vals):]
    w = w / w.sum()
    return float((vals*w).sum())

def score_buildings(gdf, insolation):
    gdf = gdf.to_crs(3857)
    gdf['area_m2'] = gdf.geometry.area
    gdf['usable_area_m2'] = gdf['area_m2'] * 0.35
    gdf['annual_potential_kwh'] = gdf['usable_area_m2'] * insolation * 365 * 0.18 * 0.75
    gdf['solar_score'] = (gdf['annual_potential_kwh'] / gdf['annual_potential_kwh'].max()).clip(0,1)
    return gdf.to_crs(4326)

def select_to_target(gdf, required_kwh):
    gdf = gdf.sort_values('solar_score', ascending=False).copy()
    gdf['cum_kwh'] = gdf['annual_potential_kwh'].cumsum()
    return gdf[gdf['cum_kwh'] <= required_kwh].copy()

# ---------------- UI Components ----------------
def render_map(gdf):
    if gdf.empty:
        st.info('No map results yet.')
        return
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13)
    for _, r in gdf.head(500).iterrows():
        folium.GeoJson(r.geometry.__geo_interface__, tooltip=f"{r.get('address','Unknown')} | {r['annual_potential_kwh']:.0f} kWh").add_to(m)
    st_folium(m, width=1200, height=700)

# ---------------- Sidebar ----------------
st.title('Urban Solar Suitability Planner')
city = st.sidebar.selectbox('City', list(CITY_TABLE_MAP.keys()))
solar_pct = st.sidebar.slider('Percent of city energy to meet with solar',1,100,30)
insolation = st.sidebar.number_input('Insolation', value=float(DEFAULT_INSOLATION[city]), step=0.1)
run = st.sidebar.button('Analyze')

if run:
    yearly = load_yearly_demand(city)
    annual_need = forecast_next_year(yearly)
    required = annual_need * solar_pct/100
    buildings = load_buildings(city)
    scored = score_buildings(buildings, insolation)
    selected = select_to_target(scored, required)
    st.session_state['yearly'] = yearly
    st.session_state['selected'] = selected
    st.session_state['annual_need'] = annual_need
    st.session_state['required'] = required

# ---------------- Main Tabs ----------------
t1,t2,t3,t4 = st.tabs(['Summary','Map','Buildings','Report'])
with t1:
    if 'annual_need' in st.session_state:
        st.metric('Forecast Annual Demand (kWh)', f"{st.session_state['annual_need']:,.0f}")
        st.metric('Solar Target (kWh)', f"{st.session_state['required']:,.0f}")
        st.line_chart(st.session_state['yearly'].set_index('Year')['annual_kwh'])
with t2:
    render_map(st.session_state.get('selected', gpd.GeoDataFrame()))
with t3:
    sel = st.session_state.get('selected', gpd.GeoDataFrame())
    if not sel.empty:
        cols = [c for c in ['address','area_m2','annual_potential_kwh','solar_score'] if c in sel.columns]
        st.dataframe(sel[cols])
with t4:
    sel = st.session_state.get('selected', gpd.GeoDataFrame())
    if not sel.empty:
        csv = sel.drop(columns='geometry').to_csv(index=False).encode()
        st.download_button('Download CSV', csv, file_name='solar_results.csv')
