from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from geopy.distance import geodesic
import osmnx as ox
import networkx as nx

from .aggregators import aggregate_drive_minutes, aggregate_context_multiplier
from .config import AppConfig


@dataclass
class TravelRV:
    mean: float
    sd: float


def estimate_drive_time_minutes(origin_lon: float, origin_lat: float, dest_lon: float, dest_lat: float, providers=None) -> TravelRV:
    """Estimate drive time using OSMnx routing if possible; fallback to speed heuristic.

    Returns a TravelRV with mean and sd (minutes).
    """
    # First try external providers if enabled
    if providers is not None:
        agg = aggregate_drive_minutes((origin_lat, origin_lon), (dest_lat, dest_lon), providers)
        if agg is not None:
            mean_min = float(agg)
            sd_min = max(5.0, 0.25 * mean_min)
            return TravelRV(mean=mean_min, sd=sd_min)

    try:
        G = ox.graph_from_point((origin_lat, origin_lon), dist=60000, network_type="drive")
        orig = ox.nearest_nodes(G, origin_lon, origin_lat)
        dest = ox.nearest_nodes(G, dest_lon, dest_lat)
        route = nx.shortest_path(G, orig, dest, weight="travel_time")
        # If travel_time not present, compute speeds
        total_seconds = 0.0
        for u, v in zip(route[:-1], route[1:]):
            data = min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", 0))
            length_m = float(data.get("length", 0.0))
            speed_kph = float(data.get("speed_kph", data.get("speed_kphs", 50)))
            if speed_kph <= 0:
                speed_kph = 50.0
            seconds = (length_m / 1000.0) / speed_kph * 3600.0
            total_seconds += seconds
        mean_min = total_seconds / 60.0
        sd_min = max(5.0, 0.25 * mean_min)
        return TravelRV(mean=mean_min, sd=sd_min)
    except Exception:
        # Straight-line heuristic with average speed 35 mph
        miles = geodesic((origin_lat, origin_lon), (dest_lat, dest_lon)).miles
        mean_min = (miles / 35.0) * 60.0
        sd_min = max(5.0, 0.30 * mean_min)
        return TravelRV(mean=mean_min, sd=sd_min)


def estimate_ride_time_minutes(*args, **kwargs) -> TravelRV:
    # rideshare similar to drive but slightly higher sd
    rv = estimate_drive_time_minutes(*args, **kwargs)
    return TravelRV(mean=rv.mean * 1.05, sd=rv.sd * 1.2)


def estimate_rail_time_minutes(origin_lon: float, origin_lat: float, dest_lon: float, dest_lat: float) -> TravelRV:
    # rougher heuristic: slower mean but higher reliability relative to drive
    miles = geodesic((origin_lat, origin_lon), (dest_lat, dest_lon)).miles
    mean_min = (miles / 30.0) * 60.0
    sd_min = max(5.0, 0.20 * mean_min)
    return TravelRV(mean=mean_min, sd=sd_min)


