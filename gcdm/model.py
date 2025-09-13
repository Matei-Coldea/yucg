from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import airports_gdf
from .travel import estimate_drive_time_minutes, estimate_ride_time_minutes, estimate_rail_time_minutes, TravelRV
from .aggregators import build_provider_config, aggregate_context_multiplier
from .components import (
    Components,
    cash_drive,
    cash_ride,
    cash_rail,
    timeval,
    risk_component,
    schedule_alignment,
    transfer_handling,
    comfort,
    carbon,
    party_vot_per_minute,
)
from .risk import gaussian_samples, risk_with_shocks
from dateutil import parser as dtparser


def compute_proc_mean_minutes(cfg: AppConfig, airport: str, luggage: str) -> float:
    s = cfg.process_times.security_mean.get(airport, 15.0)
    c = cfg.process_times.bag_drop_mean.get(luggage, 0.0)
    w = cfg.process_times.walk_mean.get(airport, 10.0)
    return float(s + c + w)


def compute_components_for_mode(cfg: AppConfig, mode: str, airport: str, dist_miles: float, access_rv: TravelRV, r_legs: int, duration_days: float, t_depart_min: float, tmin_min: float, tmax_min: float, luggage: str) -> Components:
    proc_mean = compute_proc_mean_minutes(cfg, airport, luggage)

    if mode == "drive":
        cash = cash_drive(cfg, airport, dist_miles, duration_days)
    elif mode == "ride":
        cash = cash_ride(cfg, airport, dist_miles, access_rv.mean)
    else:
        cash = cash_rail(cfg, airport)

    tv = timeval(cfg, r_legs, access_rv.mean, proc_mean)
    rk = risk_component(cfg, r_legs, access_rv, proc_mean)
    sch = schedule_alignment(cfg, airport, t_depart_min, access_rv, tmin_min, tmax_min)
    xf = transfer_handling(cfg, airport, luggage == "checked", r_legs)
    cmf = comfort(cfg, r_legs, access_rv.mean)
    crb = carbon(cfg, mode, dist_miles)
    return Components(cash=cash, timeval=tv, risk=rk, sched=sch, xfer=xf, comfort=cmf, carbon=crb)


def softmin(values: List[float], mu: float) -> float:
    x = np.array(values, dtype=float)
    return float(-1.0 / mu * np.log(np.sum(np.exp(-mu * x))))


def generalized_cost_delta(cfg: AppConfig, origins_df: pd.DataFrame, airports: List[str], modes: List[str], luggage: str = "none", r_legs: int = 1, duration_days: float = 3.0) -> pd.DataFrame:
    # Prepare airports
    ap = airports_gdf(airports).to_crs(4326)
    apts = {row.code: (row.geometry.x, row.geometry.y) for _, row in ap.iterrows()}

    rows = []
    providers = build_provider_config(cfg.__dict__.get("providers").__dict__ if hasattr(cfg, "providers") else {})
    for _, row in origins_df.iterrows():
        lon, lat, zip_code = float(row["lon"]), float(row["lat"]), str(row["zip"]) if "zip" in row else ""
        per_airport_gc: Dict[str, float] = {}
        per_airport_breakdown: Dict[str, Dict[str, float]] = {}

        for a in airports:
            alon, alat = apts[a]
            dist_miles = np.hypot((lon - alon) * 54.6, (lat - alat) * 69.0)  # rough miles
            # compute access rv by mode
            mode_rv = {}
            mode_rv["drive"] = estimate_drive_time_minutes(lon, lat, alon, alat, providers=providers)
            mode_rv["ride"] = estimate_ride_time_minutes(lon, lat, alon, alat)
            mode_rv["rail"] = estimate_rail_time_minutes(lon, lat, alon, alat)

            # Parse schedule to minutes difference relative to depart time
            try:
                t_depart = dtparser.isoparse(cfg.schedule.t_depart_iso)
                tmin_dt = dtparser.isoparse(cfg.schedule.curb_windows[a].tmin)
                tmax_dt = dtparser.isoparse(cfg.schedule.curb_windows[a].tmax)
                t_depart_min = 0.0
                tmin_min = (tmin_dt - t_depart).total_seconds() / 60.0
                tmax_min = (tmax_dt - t_depart).total_seconds() / 60.0
            except Exception:
                t_depart_min = 0.0
                tmin_min = 60.0
                tmax_min = 120.0

            # Context multiplier from providers (weather/events near origin)
            k_ctx = aggregate_context_multiplier((lon, lat), providers)

            per_mode_gc = {}
            per_mode_comp = {}
            for m in modes:
                comp = compute_components_for_mode(
                    cfg,
                    m,
                    a,
                    dist_miles,
                    mode_rv[m],
                    r_legs,
                    duration_days,
                    t_depart_min,
                    tmin_min,
                    tmax_min,
                    luggage,
                )

                # Recompute risk with shocks and context multiplier using samples
                A = gaussian_samples(mode_rv[m].mean, mode_rv[m].sd)
                P = gaussian_samples(compute_proc_mean_minutes(cfg, a, luggage), max(2.0, 0.2 * compute_proc_mean_minutes(cfg, a, luggage)))
                risk_delta = risk_with_shocks(cfg, A, P) * party_vot_per_minute(cfg) * r_legs
                risk_delta *= k_ctx
                # Replace original risk contribution
                comp_adj_total = comp.cash + comp.timeval + risk_delta + comp.sched + comp.xfer + comp.comfort + comp.carbon
                per_mode_comp[m] = comp
                per_mode_gc[m] = comp_adj_total

            gc_soft = softmin(list(per_mode_gc.values()), cfg.softmin.mu)
            per_airport_gc[a] = gc_soft
            per_airport_breakdown[a] = {
                f"{m}_total": per_mode_gc[m] for m in per_mode_gc
            }

        hvn = per_airport_gc.get("HVN", np.nan)
        others = [per_airport_gc[x] for x in per_airport_gc if x != "HVN"]
        best_other = float(np.nanmin(others)) if len(others) else np.nan
        delta = hvn - best_other if (np.isfinite(hvn) and np.isfinite(best_other)) else np.nan
        rows.append({"zip": zip_code, "lon": lon, "lat": lat, "delta_gc": delta, **{f"GC_{k}": v for k, v in per_airport_gc.items()}})

    return pd.DataFrame(rows)


