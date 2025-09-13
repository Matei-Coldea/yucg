from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import AppConfig
from .travel import TravelRV
from .risk import cvar_minus_mean, gaussian_samples, apply_context_multiplier


@dataclass
class Components:
    cash: float
    timeval: float
    risk: float
    sched: float
    xfer: float
    comfort: float
    carbon: float

    def total(self) -> float:
        return self.cash + self.timeval + self.risk + self.sched + self.xfer + self.comfort + self.carbon


def party_vot_per_minute(cfg: AppConfig) -> float:
    vot = (cfg.party.NA * cfg.party.vA_usd_per_hour + cfg.party.NC * cfg.party.vC_usd_per_hour) / 60.0
    return vot


def cash_drive(cfg: AppConfig, airport: str, distance_miles: float, duration_days: float) -> float:
    dcfg = cfg.modes.drive
    daily = dcfg.parking_daily[airport]
    weekly = dcfg.parking_weekly[airport]
    D = max(1.0, duration_days)
    park = min(np.ceil(D) * daily, np.ceil(D / 7.0) * weekly)
    vehop = dcfg.cost.cmile * 2.0 * distance_miles
    tolls = 0.0  # placeholder; extend with actual toll model
    cash = park + tolls + vehop + dcfg.cost.rem_lot_fee + dcfg.cost.shuttle_fee + dcfg.cost.fare_risk
    return float(cash)


def cash_ride(cfg: AppConfig, airport: str, distance_miles: float, mean_minutes: float) -> float:
    rcfg = cfg.modes.ride
    s = rcfg.surge
    cost = s * (rcfg.base_fare + rcfg.per_mile * 2.0 * distance_miles + rcfg.per_minute * 2.0 * mean_minutes + rcfg.booking_fee) + rcfg.fare_risk
    tip = rcfg.tip_fraction * cost
    return float(cost + tip)


def cash_rail(cfg: AppConfig, airport: str) -> float:
    rail = cfg.modes.rail
    fa = rail.adult_fare_default * cfg.party.NA
    fc = rail.child_fare_default * cfg.party.NC
    xfer = (cfg.party.NA + cfg.party.NC) * (rail.transfer_fare + rail.airtrain.get(airport, 0.0))
    return float(fa + fc + xfer + rail.station_parking)


def _tod_multiplier(cfg: AppConfig, access_mean_min: float) -> float:
    # crude: weight some portion of time as peak/night based on peak_hours
    # For complexity, assume 40% peak, 10% night, rest off-peak
    bands = {b.name: b.minutes_value for b in cfg.vot.tod_bands}
    peak = bands.get("peak", party_vot_per_minute(cfg))
    off = bands.get("offpeak", party_vot_per_minute(cfg))
    night = bands.get("night", party_vot_per_minute(cfg))
    return 0.4 * peak + 0.5 * off + 0.1 * night


def timeval(cfg: AppConfig, r: int, access_mean_min: float, proc_mean_min: float) -> float:
    # time-of-day aware valuation using configured band minute values (already USD/min)
    vot_per_min = _tod_multiplier(cfg, access_mean_min)
    return vot_per_min * r * (access_mean_min + proc_mean_min)


def risk_component(cfg: AppConfig, r: int, access_rv: TravelRV, proc_mean_min: float) -> float:
    alpha = cfg.risk.alpha
    rho = cfg.risk.rho
    k = apply_context_multiplier(1.0, cfg.risk.multipliers.weather, cfg.risk.multipliers.event, cfg.risk.multipliers.construction)

    A = gaussian_samples(access_rv.mean, access_rv.sd)
    P = gaussian_samples(proc_mean_min, max(2.0, 0.2 * proc_mean_min))
    deltaA = cvar_minus_mean(A, alpha)
    deltaP = cvar_minus_mean(P, alpha)
    return float(k * rho * party_vot_per_minute(cfg) * r * (deltaA + deltaP))


def schedule_alignment(cfg: AppConfig, airport: str, t_depart_min: float, access_rv: TravelRV, tmin_min: float, tmax_min: float, thetaE: float = 1.0, thetaL: float = 2.0) -> float:
    # U = tdepart - Ai,a,m in minutes space
    # Approximate by sampling Ai and computing penalties
    A = gaussian_samples(access_rv.mean, access_rv.sd)
    U = t_depart_min - A
    early_pen = np.maximum(tmin_min - U, 0.0)
    late_pen = np.maximum(U - tmax_min, 0.0)
    phi = thetaE * early_pen + thetaL * late_pen
    return party_vot_per_minute(cfg) * float(np.mean(phi))


def transfer_handling(cfg: AppConfig, airport: str, has_checked_bag: bool, r: int) -> float:
    # Rail miss penalty simplified
    rail = cfg.modes.rail
    # miss probability increases with negative slack; use slope parameter
    miss_prob = np.clip(rail.miss_prob_base + rail.miss_slope * 5.0, 0.0, 0.9)
    xfer_rail = (cfg.party.NA + cfg.party.lambda_child_time_weight * cfg.party.NC) * party_vot_per_minute(cfg) * miss_prob * rail.headway_minutes
    bag_pen = 0.0
    if has_checked_bag:
        bag_pen = party_vot_per_minute(cfg) * (10.0 + cfg.risk.rho * 5.0)
    geom_pen = party_vot_per_minute(cfg) * r * (5.0)
    return float(xfer_rail + bag_pen + geom_pen)


def comfort(cfg: AppConfig, r: int, access_mean_min: float, peak_fraction: float = 0.4, night_fraction: float = 0.1) -> float:
    omega = (cfg.comfort.omega_peak - 1.0) * peak_fraction + (cfg.comfort.omega_night - 1.0) * night_fraction
    return party_vot_per_minute(cfg) * r * access_mean_min * omega


def carbon(cfg: AppConfig, mode: str, distance_miles: float) -> float:
    if not cfg.carbon.enabled:
        return 0.0
    SC = cfg.carbon.social_cost_per_ton
    em = cfg.carbon.emissions_kg_per_mile.get(mode, 0.0)
    return SC / 1000.0 * em * 2.0 * distance_miles


