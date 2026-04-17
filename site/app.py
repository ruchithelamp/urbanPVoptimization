# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from supabase import create_client
import folium
from streamlit_folium import st_folium

st.set_page_config(layout='wide', page_title='Urban Solar Suitability Planner')

DATA_DIR = 'data'
CITY_TABLE_MAP = {
    'Ann Arbor':'Ann_Arbor_demand','Homestead':'Homestead_demand','Los Angeles':'LA_demand',
    'Portland':'Portland_demand','Seattle':'Seattle_demand','Tacoma':'Tacoma_demand',
    'Tallahassee':'Tallahassee_demand','Tampa':'Tampa_demand','Tucson':'TEPC_demand'
}
DEFAULT_INSOLATION = {'Ann Arbor':4.0,'Homestead':4.4,'Los Angeles':4.6,'Portland':3.0,'Seattle':2.9,'Tacoma':3.0,'Tallahassee':3.96,'Tampa':4.0,'Tucson':6.0}

supabase = create_client(st.secrets['SUPABASE_URL'], st.secrets['SUPABASE_ANON_KEY'])

# ---------------- Data Layer ----------------
@st.cache_data(ttl=3600)
def load_yearly(city):
    table = CITY_TABLE_MAP[city]
    res = supabase.table(table).select('Year,MW').limit(1000000).execute()
    df = pd.DataFrame(res.data)
    df['MW'] = pd.to_numeric(df['MW'], errors='coerce')
    out = df.groupby('Year', as_index=False)['MW'].sum()
    out['annual_kwh'] = out['MW'] * 250
    return out.sort_values('Year')

@st.cache_data(ttl=86400)
def load_buildings(city):
    path = f"{DATA_DIR}/{city.replace(' ','_')}.parquet"
    return gpd.read_parquet(path)

def forecast(df):
    vals = df.tail(3)['annual_kwh'].to_numpy()
    w = np.array([0.2,0.3,0.5])[-len(vals):]
    w = w / w.sum()
    return float((vals*w).sum())

def run_selection(gdf, required, insolation):
    gdf = gdf.copy()
    gdf['usable_area_m2'] = gdf['area_m2'] * np.where(gdf['is_residential'],0.25,0.50)
    gdf['annual_potential_kwh'] = gdf['usable_area_m2'] * insolation * 365 * 0.18 * 0.75
    gdf['solar_score'] = gdf['annual_potential_kwh'] / gdf['annual_potential_kwh'].max()
    gdf = gdf.sort_values('solar_score', ascending=False)
    gdf['cum_kwh'] = gdf['annual_potential_kwh'].cumsum()
    return gdf[gdf['cum_kwh'] <= required]

def draw_map(gdf):
    if gdf.empty:
        st.info('No selected buildings.')
        return
    sample = gdf.head(300)
    center = [sample.geometry.centroid.y.mean(), sample.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13)
    for _, r in sample.iterrows():
        folium.GeoJson(r.geometry.__geo_interface__, tooltip=f"{r['address']} | {r['annual_potential_kwh']:.0f} kWh").add_to(m)
    st_folium(m, width=1200, height=700)

st.title('Urban Solar Suitability Planner')
city = st.sidebar.selectbox('City', list(CITY_TABLE_MAP.keys()))
solar_pct = st.sidebar.slider('Percent of city energy to meet with solar',1,100,30)
insolation = st.sidebar.number_input('Insolation', value=float(DEFAULT_INSOLATION[city]), step=0.1)

if st.sidebar.button('Analyze'):
    yearly = load_yearly(city)
    demand = forecast(yearly)
    required = demand * solar_pct / 100
    bld = load_buildings(city)
    sel = run_selection(bld, required, insolation)
    st.session_state['yearly']=yearly
    st.session_state['demand']=demand
    st.session_state['required']=required
    st.session_state['sel']=sel

A,B,C,D = st.tabs(['Summary','Map','Buildings','Export'])
with A:
    if 'demand' in st.session_state:
        st.metric('Forecast Demand (kWh)', f"{st.session_state['demand']:,.0f}")
        st.metric('Solar Target (kWh)', f"{st.session_state['required']:,.0f}")
        st.line_chart(st.session_state['yearly'].set_index('Year')['annual_kwh'])
with B:
    draw_map(st.session_state.get('sel', gpd.GeoDataFrame()))
with C:
    sel = st.session_state.get('sel', gpd.GeoDataFrame())
    if not sel.empty:
        st.dataframe(sel[['address','is_residential','area_m2','annual_potential_kwh','solar_score']])
with D:
    sel = st.session_state.get('sel', gpd.GeoDataFrame())
    if not sel.empty:
        csv = sel.drop(columns='geometry').to_csv(index=False).encode()
        st.download_button('Download CSV', csv, file_name='solar_results.csv')
