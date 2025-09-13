from __future__ import annotations

import os
from typing import Optional

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import folium
from .data import airports_gdf


def choropleth_delta(zctas: gpd.GeoDataFrame, results: pd.DataFrame, title: str, outdir: str, crs: int = 3857, interactive: bool = True) -> None:
    os.makedirs(outdir, exist_ok=True)
    gdf = zctas.merge(results[["zip", "delta_gc"]], on="zip", how="left")
    gdf = gdf.to_crs(crs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    vmin, vmax = np.nanpercentile(gdf["delta_gc"], [5, 95]) if gdf["delta_gc"].notna().any() else (0, 1)
    gdf.plot(column="delta_gc", cmap="RdBu_r", scheme="quantiles", k=7, legend=True, vmin=vmin, vmax=vmax, ax=ax, missing_kwds={"color": "lightgray", "label": "No data"})
    ax.set_axis_off()
    ax.set_title(title)
    try:
        ctx.add_basemap(ax, crs=gdf.crs.to_string())
    except Exception:
        pass
    png_path = os.path.join(outdir, "gcdm_delta.png")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    if interactive:
        gdf_ll = gdf.to_crs(4326)
        center = [gdf_ll.geometry.centroid.y.mean(), gdf_ll.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=8, tiles="cartodbpositron")
        chor = folium.Choropleth(
            geo_data=gdf_ll.to_json(),
            data=gdf_ll,
            columns=["zip", "delta_gc"],
            key_on="feature.properties.zip",
            fill_color="RdBu",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Delta GC (HVN - min others)",
        )
        chor.add_to(m)
        # Attach popups with delta
        folium.GeoJson(
            gdf_ll,
            name="ZCTAs",
            style_function=lambda x: {"fillOpacity": 0},
            tooltip=folium.features.GeoJsonTooltip(fields=["zip", "delta_gc"], aliases=["ZIP", "Delta GC"], localize=True),
        ).add_to(m)
        # Add airport markers and origin centroids
        aps = airports_gdf(["HVN", "JFK", "LGA", "EWR"]).to_crs(4326)
        for _, r in aps.iterrows():
            folium.Marker([r.geometry.y, r.geometry.x], tooltip=f"{r['code']} - {r['name']}", icon=folium.Icon(color="blue", icon="plane", prefix="fa")).add_to(m)
        for _, r in results.merge(zctas[["zip", "geometry"]], on="zip", how="left").to_crs(4326).iterrows():
            g = r.geometry
            if g is not None and hasattr(g, "centroid"):
                folium.CircleMarker([g.centroid.y, g.centroid.x], radius=2, color="#555", fill=True, fill_opacity=0.7).add_to(m)
        html_path = os.path.join(outdir, "gcdm_delta.html")
        m.save(html_path)


