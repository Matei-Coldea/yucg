from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class ProviderError(Exception):
    pass


@dataclass
class ProviderConfig:
    enable_google: bool
    enable_openrouteservice: bool
    enable_weather: bool
    enable_events: bool
    enable_mapbox: bool
    google_api_key: str
    ors_api_key: str
    owm_api_key: str
    ticketmaster_api_key: str
    mapbox_api_key: str


def build_provider_config(raw: Dict[str, Any]) -> ProviderConfig:
    return ProviderConfig(
        enable_google=bool(raw.get("enable_google", False)),
        enable_openrouteservice=bool(raw.get("enable_openrouteservice", False)),
        enable_weather=bool(raw.get("enable_weather", False)),
        enable_events=bool(raw.get("enable_events", False)),
        enable_mapbox=bool(raw.get("enable_mapbox", False)),
        google_api_key=str(raw.get("google_api_key", "")),
        ors_api_key=str(raw.get("ors_api_key", "")),
        owm_api_key=str(raw.get("owm_api_key", "")),
        ticketmaster_api_key=str(raw.get("ticketmaster_api_key", "")),
        mapbox_api_key=str(raw.get("mapbox_api_key", "")),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True)
def google_drive_minutes(origin: tuple[float, float], dest: tuple[float, float], api_key: str) -> Optional[float]:
    try:
        import googlemaps  # type: ignore

        gmaps = googlemaps.Client(key=api_key)
        resp = gmaps.distance_matrix(origins=[origin], destinations=[dest], mode="driving", departure_time=dt.datetime.now())
        rows = resp.get("rows", [])
        if not rows:
            return None
        elems = rows[0].get("elements", [])
        if not elems:
            return None
        dur = elems[0].get("duration_in_traffic") or elems[0].get("duration")
        if not dur:
            return None
        seconds = float(dur.get("value", 0.0))
        return seconds / 60.0
    except Exception as e:
        raise ProviderError(str(e))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True)
def ors_drive_minutes(origin: tuple[float, float], dest: tuple[float, float], api_key: str) -> Optional[float]:
    try:
        import openrouteservice  # type: ignore

        client = openrouteservice.Client(key=api_key)
        route = client.directions(coordinates=[(origin[1], origin[0]), (dest[1], dest[0])], profile="driving-car", format="geojson")
        feats = route.get("features", [])
        if not feats:
            return None
        props = feats[0].get("properties", {})
        summary = props.get("summary", {})
        seconds = float(summary.get("duration", 0.0))
        return seconds / 60.0
    except Exception as e:
        raise ProviderError(str(e))


def weather_context_multiplier(lat: float, lon: float, api_key: str) -> float:
    try:
        import pyowm  # type: ignore

        owm = pyowm.OWM(api_key)
        mgr = owm.weather_manager()
        obs = mgr.weather_at_coords(lat, lon)
        w = obs.weather
        wind = w.wind().get("speed", 0.0)
        rain = 1.0 if w.rain else 0.0
        snow = 1.0 if w.snow else 0.0
        vis = w.visibility_distance or 10000
        # Compose a multiplier roughly scaling congestion and variability
        mult = 1.0 + 0.02 * wind + 0.3 * rain + 0.5 * snow + (0.2 if vis < 2000 else 0.0)
        return float(np.clip(mult, 1.0, 2.0))
    except Exception:
        return 1.0


def events_context_multiplier(lat: float, lon: float, api_key: str) -> float:
    # Very rough heuristic; if there are many events nearby in next 24h, bump multiplier
    try:
        import requests

        url = "https://app.ticketmaster.com/discovery/v2/events.json"
        params = {"apikey": api_key, "latlong": f"{lat},{lon}", "radius": 30, "unit": "miles"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        total = int(data.get("page", {}).get("totalElements", 0))
        if total >= 100:
            return 1.3
        if total >= 50:
            return 1.15
        if total >= 10:
            return 1.05
        return 1.0
    except Exception:
        return 1.0


def aggregate_drive_minutes(origin: tuple[float, float], dest: tuple[float, float], providers: ProviderConfig) -> Optional[float]:
    vals = []
    if providers.enable_google and providers.google_api_key:
        try:
            v = google_drive_minutes(origin, dest, providers.google_api_key)
            if v:
                vals.append(v)
        except ProviderError:
            pass
    if providers.enable_openrouteservice and providers.ors_api_key:
        try:
            v = ors_drive_minutes(origin, dest, providers.ors_api_key)
            if v:
                vals.append(v)
        except ProviderError:
            pass
    if providers.enable_mapbox and providers.mapbox_api_key:
        try:
            v = mapbox_drive_minutes(origin, dest, providers.mapbox_api_key)
            if v:
                vals.append(v)
        except ProviderError:
            pass

    if not vals:
        return None
    # Return a conservative blend (upper quartile)
    return float(np.quantile(np.array(vals, dtype=float), 0.75))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True)
def mapbox_drive_minutes(origin: tuple[float, float], dest: tuple[float, float], api_key: str) -> Optional[float]:
    try:
        import requests

        base = "https://api.mapbox.com/directions/v5/mapbox/driving"
        # Mapbox expects lon,lat order
        coords = f"{origin[1]},{origin[0]};{dest[1]},{dest[0]}"
        params = {"alternatives": "false", "geometries": "geojson", "overview": "simplified", "access_token": api_key}
        r = requests.get(f"{base}/{coords}", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        routes = data.get("routes", [])
        if not routes:
            return None
        seconds = float(routes[0].get("duration", 0.0))
        return seconds / 60.0
    except Exception as e:
        raise ProviderError(str(e))


def aggregate_context_multiplier(origin: tuple[float, float], providers: ProviderConfig) -> float:
    lat, lon = origin[1], origin[0]
    mults = [1.0]
    if providers.enable_weather and providers.owm_api_key:
        mults.append(weather_context_multiplier(lat, lon, providers.owm_api_key))
    if providers.enable_events and providers.ticketmaster_api_key:
        mults.append(events_context_multiplier(lat, lon, providers.ticketmaster_api_key))
    return float(np.prod(mults))


