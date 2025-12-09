# solar_suitability.py

import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
from supabase import create_client, Client
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon

class SolarSuitability:

    PANEL_EFFICIENCY = 0.18
    PERFORMANCE_RATIO = 0.75
    DEFAULT_INSOLATION = {"Ann Arbor": 4.0, "Tucson": 6.0}

    def __init__(self, city, supabase_url, supabase_key,
                 solar_pct, commercial_pct, insolation_override):

        self.city = city
        self.solar_pct = solar_pct
        self.commercial_pct = commercial_pct
        self.insolation_override = insolation_override

        self.supabase = create_client(supabase_url, supabase_key)

    def load_demand_table(self):
        table_name = "Ann_Arbor_demand" if self.city == "Ann Arbor" else "TEPC_demand"
        res = self.supabase.table(table_name).select("*").limit(100000).execute()

        df = pd.DataFrame(res.data)

        return df

    def fetch_buildings_osm(self):
        place_name = f"{self.city}, USA"

        residential_tags = {
            "building": [
                "apartments", "residential", "garage", "detached",
                "bungalow", "house", "semidetached_house"
            ]
        }

        commercial_tags = {
            "building": [
                "office", "university", "yes", "train_station", "courthouse",
                "hospital", "industrial", "warehouse", "hotel", "commercial"
            ]
        }

        r_gdf = ox.features.features_from_place(place_name, tags=residential_tags)
        c_gdf = ox.features.features_from_place(place_name, tags=commercial_tags)

        gdf = pd.concat([r_gdf, c_gdf])

        drop = {"museum", "stable", "service", "greenhouse", "kindergarten"}
        gdf = gdf[~gdf["building"].isin(drop)]
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

        # Residential flag
        gdf["is_residential"] = gdf["building"].isin(residential_tags["building"])

        # Area
        gdf_proj = ox.projection.project_gdf(gdf)
        gdf["area_m2"] = gdf_proj.area

        return gdf

    def compute_city_annual_kwh(self, df):
        """
        Convert 15-min MW demand to annual kWh and estimate next year's demand.
        """
        summary = df.groupby("Year")["MW"].sum().reset_index()
        summary["annual_kWh"] = summary["MW"] * 0.25 * 1000
        summary["growth_rate"] = summary["annual_kWh"].pct_change()

        last = summary.iloc[-1]

        growth = 0.02 if pd.isna(last["growth_rate"]) else last["growth_rate"]
        return last["annual_kWh"] * (1 + growth)

    def polygon_orientation(self, poly: Polygon):
        if poly.is_empty:
            return np.nan
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)

        best_len = 0
        best_angle = 0
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L = np.hypot(dx, dy)

            if L > best_len:
                best_len = L
                best_angle = np.degrees(np.arctan2(dy, dx))

        return (90 - best_angle) % 360

    def estimate_tilt_from_height_and_span(self, gdf):
        """
        Estimate roof tilt using height / levels.
        """
        if "height" in gdf.columns:
            heights = pd.to_numeric(gdf["height"], errors="coerce")
        elif "building:levels" in gdf.columns:
            heights = pd.to_numeric(gdf["building:levels"], errors="coerce") * 3.2
        else:
            heights = pd.Series(np.nan, index=gdf.index)

        def approx_width(geom):
            try:
                mrr = geom.minimum_rotated_rectangle
                perim = mrr.length
                return max(perim / 4, np.sqrt(geom.area))
            except Exception:
                return np.sqrt(geom.area)

        #gdf["width_m"] = gdf.geometry.to_crs(epsg=3857).apply(lambda geom: approx_width(geom)) change to keep crs projection?
        gdf["width_m"] = gdf.geometry.apply(approx_width)
        gdf["tilt_deg"] = np.degrees(np.arctan2(heights.fillna(5), gdf["width_m"] / 2))
        gdf["tilt_deg"] = gdf["tilt_deg"].clip(3, 45)
        return gdf

    def orientation_match_score(self, roof_deg, ideal_deg):
        diff = abs(roof_deg - ideal_deg) % 360
        diff = diff if diff <= 180 else 360 - diff
        return 1 - (diff / 180)

    def compute_suitability_scores(self, gdf, irr_factor, ideal_azimuth):
        gdf["orientation_deg"] = gdf.geometry.apply(self.polygon_orientation)
        gdf["orientation_score"] = gdf["orientation_deg"].apply(
            lambda a: self.orientation_match_score(a, ideal_azimuth)
        )

        gdf["tilt_score"] = 1 - (abs(gdf["tilt_deg"] - 25) / 40)
        gdf["tilt_score"] = gdf["tilt_score"].clip(0, 1)

        gdf["exposure_score"] = 1 - gdf["shade"] #clip(0,1) again?

        irr = np.clip(irr_factor, 0, 1)

        gdf["solar_score"] = (
            0.35 * gdf["orientation_score"]
            + 0.25 * gdf["tilt_score"]
            + 0.25 * gdf["exposure_score"]
            + 0.15 * irr
        )
        #gdf["solar_score"] = gdf["solar_score"].clip(0,1)
        return gdf

    def estimate_building_annual_potential(self, gdf):
        COMMERCIAL_FRACTION = 0.50
        RESIDENTIAL_FRACTION = 0.25

        gdf["usable_fraction"] = gdf["is_residential"].apply(
            lambda x: RESIDENTIAL_FRACTION if x else COMMERCIAL_FRACTION
        )
        gdf["usable_area_m2"] = gdf["area_m2"] * gdf["usable_fraction"]

        daily = self.insolation_override
        factor = daily * 365 * self.PANEL_EFFICIENCY * self.PERFORMANCE_RATIO
        gdf["annual_potential_kwh"] = gdf["usable_area_m2"] * factor

        return gdf

    def select_buildings(self, gdf, required_kwh, commercial_pct):
        comm = gdf[~gdf["is_residential"]].sort_values("solar_score", ascending=False) #.copy()?
        resid = gdf[gdf["is_residential"]].sort_values("solar_score", ascending=False) #.copy()?

        sel = []
        total_kwh = 0
        ptr_c = ptr_r = 0
        selected_comm = 0

        target_frac = commercial_pct / 100

        while total_kwh < required_kwh and (ptr_c < len(comm) or ptr_r < len(resid)):

            curr_frac = selected_comm / len(sel) if sel else 0 #if sel > 0 else 0?

            if curr_frac < target_frac and ptr_c < len(comm):
                row = comm.iloc[ptr_c]; ptr_c += 1
            elif ptr_r < len(resid):
                row = resid.iloc[ptr_r]; ptr_r += 1
            elif ptr_c < len(comm):
                row = comm.iloc[ptr_c]; ptr_c += 1
            else:
                break

            sel.append(row)
            total_kwh += row["annual_potential_kwh"]
            if not row["is_residential"]:
                selected_comm += 1

        selected_gdf = gpd.GeoDataFrame(sel, crs=gdf.crs)
        actual_pct = (selected_comm / len(sel) * 100) if sel else 0

        return selected_gdf, total_kwh, len(sel), actual_pct
