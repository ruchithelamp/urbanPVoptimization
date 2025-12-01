import streamlit as st
import pandas as pd
import psycopg2

db_url = st.secrets["connections"]["supabase"]



st.title("EIA Hourly Demand from supabase db")

# connection to db 
conn = psycopg2.connect(db_url)

# Simple query
query = "SELECT * FROM TEPC_demand LIMIT 3"

df = pd.read_sql(query, conn)

st.dataframe(df)

conn.close()

#df1 = pd.DataFrame({
#    'first column': ['Ann Arbor', 'Tucson'],
#    })

#option = st.selectbox(
#    'Select city',
#    df1['first column'])

# Map container
#map_container = st.container(border=True)
#with map_container:
#    st.map(data=df1.rename(columns={'first column': 'city'}).assign(lat=[42.2808, 32.2226], lon=[-83.7430, -110.9747]))

# City specs container
#city_specs = "City specs." * 1000

#with st.container(height=300):
#    st.markdown(city_specs)

#st.slider("Select Power Percentage", 0, 100, 0)

#st.slider("Select Commercial Coverage", 0, 100, 0)

# Results container
#results_container = "Results container." * 1000
#with st.container(height=300):
#    st.markdown(results_container)

#conn.close()