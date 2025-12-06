# solar_suitability.py

import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
from supabase import create_client, Client
from shapely.geometry import Polygon

class SolarSuitabilityPlanner:
    def __init__(self, city: str, solar_pct: float, commercial_pct: float, insolation_override: float):
        self.city = city
        self.solar_pct = solar_pct
        self.commercial_pct = commercial_pct
        self.insolation_override = insolation_override
        
        # Constants
        self.DEFAULT_INSOLATION = {"Ann Arbor": 4.0, "Tucson": 6.0}
        self.PANEL_EFFICIENCY = 0.18
        self.PERFORMANCE_RATIO = 0.75
        self.url = "YOUR_SUPABASE_URL"
        self.key = "YOUR_SUPABASE_ANON_KEY"
        self.supabase = create_client(self.url, self.key)
        
        # Streamlit configuration
        # st.set_page_config(layout="wide", page_title="Solar Suitability Planner")

    @st.cache_data(show_spinner=False)
    def load_demand_table(self):
        """Load city demand table from Supabase."""
        table_name = "Ann_Arbor_demand" if self.city == "Ann Arbor" else "TEPC_demand"
        res = self.supabase.table(table_name).select("*").limit(100000).execute()
        data = res.data
        df = pd.DataFrame(data)
        
        if 'MW' in df.columns:
            df['MW'] = pd.to_numeric(df['MW'], errors='coerce')
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        return df

    def fetch_buildings_osm(self, place_name: str):
        """Fetch building footprints using OSMnx."""
        tags = {"building": ['apartments', 'residential', 'garage', 'detached', 'bungalow', 'house', 'semidetached_house']}
        r_gdf = ox.features.features_from_place(place_name, tags=tags)

        tags = {"building": ['office', 'university', 'yes',  'train_station', 'courthouse', 'hospital', 'industrial']}
        c_gdf = ox.features.features_from_place(place_name, tags=tags)

        gdf = pd.concat([r_gdf, c_gdf])

        drop_labels = ['museum', 'stable', 'service', 'greenhouse', 'kindergarten', 'hangar', 'static_caravan', 
                       'collapsed', 'ses', 'bunker', 'kiosk', 'no', 'sports_centre', 'chapel', 'historic', 'grandstand']
        
        gdf = gdf[~gdf["building"].isin(drop_labels)]
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        gdf["is_residential"] = gdf.apply(lambda row: row["building"].lower() in ['apartments', 'residential', 'garage'], axis=1)
        gdf_proj = ox.projection.project_gdf(gdf)
        gdf['area_m2'] = gdf_proj.area
        
        return gdf

    def compute_city_annual_kwh(self, df: pd.DataFrame):
        """Compute the annual kWh for the city."""
        summary = df.groupby('Year')['MW'].sum().reset_index(name='total_MW_sum')
        summary['annual_kWh'] = summary['total_MW_sum'] * 0.25 * 1000
        summary = summary.sort_values("Year").reset_index(drop=True)
        summary["growth_rate"] = summary["annual_kWh"].pct_change()
        
        last_year = summary.iloc[-1]["Year"]
        last_demand = summary.iloc[-1]["annual_kWh"]
        last_growth = summary.iloc[-1]["growth_rate"]
        
        if pd.isna(last_growth):
            last_growth = 0.02
        
        predicted = last_demand * (1 + last_growth)
        return predicted

    def compute_suitability_scores(self, gdf: gpd.GeoDataFrame, irradiance_factor: float, ideal_sun_azimuth: float):
        """Compute solar suitability scores for each building."""
        gdf["orientation_deg"] = gdf.geometry.apply(lambda geom: polygon_orientation(geom))
        gdf["orientation_score"] = gdf["orientation_deg"].apply(lambda a: orientation_match_score(a, ideal_sun_azimuth))
        
        gdf["tilt_score"] = 1.0 - (np.abs(gdf["tilt_deg"] - 25.0) / 40.0)
        gdf["tilt_score"] = gdf["tilt_score"].clip(0, 1)
        
        gdf["exposure_score"] = (1.0 - gdf["shade"]).clip(0, 1)
        irr = np.clip(irradiance_factor, 0.0, 1.0)
        
        gdf["solar_score"] = (
            0.35 * gdf["orientation_score"] +
            0.25 * gdf["tilt_score"] +
            0.25 * gdf["exposure_score"] +
            0.15 * irr
        )
        gdf["solar_score"] = gdf["solar_score"].clip(0, 1)
        return gdf

    def analyze(self):
        """Run the full analysis."""
        # Full analysis code...
        pass
