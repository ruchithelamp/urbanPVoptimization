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

df1 = pd.DataFrame({
    'first column': ['Ann Arbor', 'Tucson'],
    })

option = st.selectbox(
    'Select city',
    df1['first column'])

st.slider("Select Power Percentage", 0, 100, 0)

st.slider("Select Commercial Coverage", 0, 100, 0)