import streamlit as st
import pandas as pd
import psycopg2

db_url = st.secrets["connections"]["supabase"]



st.title("EIA Hourly Demand from supabase db")


# connection to db 
conn = psycopg2.connect(db_url)

# Simple query
query = "SELECT * FROM eia_hourly_demand LIMIT 3"

df = pd.read_sql(query, conn)

st.dataframe(df)

df1 = pd.DataFrame({
    'first column': ['Ann Arbor', 'Tucson'],
    })

option = st.selectbox(
    'Select city',
    df1['first column'])

#st.write(x, 'Select Power Percentage', x * x)
st.slider(label, min_value=0, max_value=100, value=None, step=1, format=%d)

st.write(y, 'Select Commercial Coverage', x * y)
y = st.slider('y')  # 👈 this is a widget

conn.close()