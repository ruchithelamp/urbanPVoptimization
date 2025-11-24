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

df1 = pd.DataFrame({
    'first column': ['Ann Arbor', 'Tucson'],
    })

option = st.selectbox(
    'Which number do you like best?',
    df1['first column'])

'Selected: ', option

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'Select Power Percentage', x * x])

y = st.slider('y')  # 👈 this is a widget
st.write(y, 'Select Commercial Coverage', x * y])

conn.close()