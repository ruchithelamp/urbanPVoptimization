# prepare_city_data.py
import os
import pandas as pd
import geopandas as gpd
import osmnx as ox

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

CITY_TAGS_RES = {'building':['apartments','residential','detached','bungalow','house','semidetached_house']}
CITY_TAGS_COM = {'building':['office','university','yes','train_station','courthouse','hospital','industrial','warehouse','hotel','commercial','retail','college','government','data_center']}

CITIES = ['Ann Arbor','Homestead','Los Angeles','Portland','Seattle','Tacoma','Tallahassee','Tampa','Tucson']

ox.settings.use_cache = True
ox.settings.timeout = 120

def fetch_city(city):
    print(f'Fetching {city}')
    r = ox.features.features_from_place(f'{city}, USA', CITY_TAGS_RES)
    c = ox.features.features_from_place(f'{city}, USA', CITY_TAGS_COM)
    gdf = pd.concat([r,c], ignore_index=False)
    gdf = gdf[gdf.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
    gdf['is_residential'] = gdf['building'].isin(CITY_TAGS_RES['building'])
    gdf = gdf.to_crs(3857)
    gdf['area_m2'] = gdf.geometry.area
    gdf = gdf[gdf['area_m2'] > 40]
    if 'addr:housenumber' in gdf.columns and 'addr:street' in gdf.columns:
        gdf['address'] = gdf['addr:housenumber'].fillna('').astype(str) + ' ' + gdf['addr:street'].fillna('').astype(str)
    else:
        gdf['address'] = ''
    gdf = gdf.to_crs(4326)
    path = os.path.join(DATA_DIR, city.replace(' ','_') + '.parquet')
    gdf.to_parquet(path)
    print(f'Saved {path} ({len(gdf)} rows)')

if __name__ == '__main__':
    for city in CITIES:
        try:
            fetch_city(city)
        except Exception as e:
            print(city, e)
if __name__ == '__main__':
