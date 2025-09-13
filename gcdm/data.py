from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


CT_BOUNDS_WKT = "POLYGON((-73.727 41.012, -73.727 42.05, -71.786 42.05, -71.786 41.012, -73.727 41.012))"


@dataclass
class Airport:
    code: str
    name: str
    lon: float
    lat: float


AIRPORTS: Dict[str, Airport] = {
    "HVN": Airport("HVN", "Tweed New Haven", -72.887, 41.263),
    "JFK": Airport("JFK", "John F. Kennedy Intl", -73.7781, 40.6413),
    "LGA": Airport("LGA", "LaGuardia", -73.8733, 40.7769),
    "EWR": Airport("EWR", "Newark Liberty Intl", -74.1745, 40.6895),
}


def airports_gdf(codes: List[str]) -> gpd.GeoDataFrame:
    rows = []
    for c in codes:
        a = AIRPORTS[c]
        rows.append({"code": a.code, "name": a.name, "geometry": Point(a.lon, a.lat)})
    gdf = gpd.GeoDataFrame(rows, crs=4326)
    return gdf


def load_ct_zctas() -> gpd.GeoDataFrame:
    """Load CT ZIP Code Tabulation Areas via geopandas built-in or census TIGER.

    This uses a simple fallback: query all ZCTAs and clip to a CT bbox.
    """
    # Try census ZCTAs via public URL (geopandas can read zipped shapefiles)
    # Using 2020 ZCTA shapefile from TIGER
    url = "https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/tl_2020_us_zcta520.zip"
    zcta = gpd.read_file(url)
    zcta = zcta.to_crs(4326)
    bbox = gpd.GeoSeries.from_wkt([CT_BOUNDS_WKT], crs=4326).iloc[0]
    zcta = zcta[zcta.geometry.intersects(bbox)]
    zcta = zcta.copy()
    zcta["ZCTA5CE20"] = zcta["ZCTA5CE20"].astype(str)
    zcta = zcta.rename(columns={"ZCTA5CE20": "zip"})[["zip", "geometry"]]
    return zcta


def ct_zip_centroids(zctas: gpd.GeoDataFrame) -> pd.DataFrame:
    cent = zctas.copy().to_crs(4326)
    cent["lon"] = cent.geometry.centroid.x
    cent["lat"] = cent.geometry.centroid.y
    return cent[["zip", "lon", "lat"]]


