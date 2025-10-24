import streamlit as st
import pandas as pd
import psycopg2

db_url = st.secrets["connections"]["supabase"]



st.title("EIA Hourly Demand from supabase db")


# connection to db 
conn = psycopg2.connect(db_url)

# Simple query
query = "SELECT * FROM eia_hourly_demand LIMIT 10"

df = pd.read_sql(query, conn)

st.dataframe(df)

conn.close()