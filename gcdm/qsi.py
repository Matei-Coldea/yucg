from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from dateutil import parser as dtparser

from .config import AppConfig, QSIConfig


@dataclass
class Itinerary:
    origin_airport: str
    hub: str
    dest: str
    mode_access: str
    block_minutes: float
    layover_minutes: float
    L_star: float
    U_star: float
    sk: float  # transfer stress factor
    tdep_iso: str
    tarr_iso: str
    cancel_rate: float
    inbound_delay_mean: float
    inbound_taxi_in_mean: float
    delay_pos_mean: float
    delay_var: float
    hub_WxRisk: float
    hub_CapacityRisk: float
    reprotect_time_mean: float
    n_alternates_window: int
    seat_pitch: float
    wifi_rel: float
    is_widebody: int
    is_priority: int
    has_lounge: int
    has_precheck: int
    price_mean: float
    price_ancillary: float
    price_rebates: float
    seats_available: int
    dupcount: int
    loadfactor_risk: float
    bag_miss_prob: float
    hub_bank_times: List[str]


def softmin_access(cfg: AppConfig, per_mode_gc: Dict[str, float]) -> float:
    mu = cfg.qsi.muA
    vals = np.array([per_mode_gc[m] for m in per_mode_gc], dtype=float)
    return float(-1.0 / mu * np.log(np.sum(np.exp(-mu * vals))))


def access_factor(cfg: AppConfig, gc_soft: float) -> float:
    return float(np.exp(-cfg.qsi.betaA * gc_soft))


def layover_quality(Lk: float, sk: float, L_star: float, U_star: float, alpha_short: float = 0.03, alpha_long: float = 0.01, alpha_s: float = 0.02) -> float:
    val = alpha_short * max(L_star - Lk, 0.0) + alpha_long * max(Lk - U_star, 0.0) + alpha_s * sk
    return val


def hod_penalty(tdep: str, tarr: str, eta_red: float = 0.1, eta_curfew: float = 0.05) -> float:
    d = dtparser.isoparse(tdep)
    a = dtparser.isoparse(tarr)
    is_redeye = a.hour < 6
    is_curfew = d.hour < 6 or d.hour > 22
    return eta_red * (1.0 if is_redeye else 0.0) + eta_curfew * (1.0 if is_curfew else 0.0)


def reliability_terms(cfg: QSIConfig, k: Itinerary, Lk: float, hub: str) -> float:
    betaD, betaV = cfg.betaD, cfg.betaV
    cancel = (1.0 - k.cancel_rate)
    # misconnect probability
    mct = cfg.hub_risks.get(hub, {}).get("MCT", 40)
    pi_mis = max(0.0, min(1.0, 1.0 - np.exp(-(max(Lk - mct, 0.0)) / 20.0)))
    rel = cancel * (1.0 - pi_mis) * np.exp(-betaD * max(k.delay_pos_mean, 0.0) - betaV * max(k.delay_var, 0.0))
    # hub risk
    Wx = cfg.hub_risks.get(hub, {}).get("WxRisk", 0.2)
    Cap = cfg.hub_risks.get(hub, {}).get("CapacityRisk", 0.3)
    rel *= np.exp(-cfg.betaW * Wx - cfg.betaC * Cap)
    return float(rel)


def reprotect(cfg: QSIConfig, k: Itinerary, hub: str) -> float:
    t = np.exp(-cfg.beta_rec * max(k.reprotect_time_mean, 0.0))
    alt = 1.0 + cfg.xi * max(k.n_alternates_window, 0)
    return float(t * alt)


def product_score(cfg: QSIConfig, k: Itinerary) -> float:
    q = cfg.quality_weights
    qk = q.wseat * k.seat_pitch + q.wwifi * k.wifi_rel + q.wac * float(k.is_widebody) + q.wprio * float(k.is_priority)
    return float(np.exp(cfg.betaQ * qk))


def perks(cfg: QSIConfig, k: Itinerary) -> float:
    return float(np.exp(cfg.beta_ffp * k.price_rebates + cfg.beta_lounge * k.has_lounge + cfg.beta_pre * k.has_precheck))


def price_availability(cfg: QSIConfig, k: Itinerary, s: str, state: dict) -> float:
    betaP = next((seg.beta_P for seg in cfg.segments if seg.name == s), 0.01)
    p_star = next((seg.p_star for seg in cfg.segments if seg.name == s), 300)
    n_req = next((seg.seats_threshold for seg in cfg.segments if seg.name == s), 2)
    eff = k.price_mean * state.get("price_multiplier", 1.0) + k.price_ancillary - k.price_rebates
    avail = 1.0 if (eff <= p_star and k.seats_available >= n_req) else 0.3
    return float(avail * np.exp(-betaP * eff))


def duplicates_scarcity(cfg: QSIConfig, k: Itinerary) -> float:
    return float(1.0 / (1.0 + max(k.dupcount, 0)) * np.exp(-cfg.beta_scar * max(k.loadfactor_risk, 0.0)))


def bank_alignment(cfg: QSIConfig, k: Itinerary, hub: str) -> float:
    arr = dtparser.isoparse(k.tarr_iso)
    bts = cfg.hub_risks.get(hub, {}).get("bank_times", [])
    if not bts:
        return 1.0
    diffs = []
    for t in bts:
        b = dtparser.parse(t)
        b = arr.replace(hour=b.hour, minute=b.minute, second=0, microsecond=0)
        diffs.append(abs((arr - b).total_seconds()) / 60.0)
    return float(np.exp(-cfg.beta_B * min(diffs) / max(cfg.bank_window_w, 1e-3)))


def iops(cfg: QSIConfig, k: Itinerary, hub: str) -> float:
    mct = cfg.hub_risks.get(hub, {}).get("MCT", 40)
    Lk = k.layover_minutes
    return float(np.exp(-cfg.beta_MCT * max(mct - Lk, 0.0)) * np.exp(-cfg.beta_bag * max(k.bag_miss_prob, 0.0)))


def itinerary_weight(cfg: AppConfig, k: Itinerary, segment: str, state: dict) -> float:
    # Schedule block and layover quality
    T = k.block_minutes + k.layover_minutes
    phi_conn = layover_quality(k.layover_minutes, k.sk, k.L_star, k.U_star)
    phi_hod = hod_penalty(k.tdep_iso, k.tarr_iso)
    # Reliability
    R = reliability_terms(cfg.qsi, k, k.layover_minutes, k.hub)
    Rh = 1.0  # included in reliability_terms for simplicity
    REC = reprotect(cfg.qsi, k, k.hub)
    # Product and perks
    Pk = product_score(cfg.qsi, k)
    ALk = perks(cfg.qsi, k)
    # Price/availability by state
    Fk = price_availability(cfg.qsi, k, segment, state)
    # Duplicates/scarcity and banks
    Dk = duplicates_scarcity(cfg.qsi, k)
    Bk = bank_alignment(cfg.qsi, k, k.hub)
    IOPS = iops(cfg.qsi, k, k.hub)
    # Combine schedule, reliability, price, product, banks
    # W = f_gamma * exp(-beta_T T) * exp(-beta_C phi_conn) * exp(-beta_H phi_hod) * R * Rh * REC * Pk * ALk * Fk * Dk * Bk * IOPS
    betaT = next((seg.beta_P for seg in cfg.qsi.segments if seg.name == segment), 0.01)  # reuse beta_P as proxy for beta_T,s
    betaC = 0.02
    betaH = 0.02
    f_gamma = 1.0  # frequency factor; assume baked in via duplicates
    W = f_gamma * np.exp(-betaT * T) * np.exp(-betaC * phi_conn) * np.exp(-betaH * phi_hod) * R * Rh * REC * Pk * ALk * Fk * Dk * Bk * IOPS
    return float(W)


def qsi_plus_for_market(cfg: AppConfig, per_mode_gc: Dict[str, float], itineraries: List[Itinerary], segments: List[str]) -> Dict[str, Any]:
    # Access factor from door->curb generalized cost across modes
    gc_soft = softmin_access(cfg, per_mode_gc)
    A = access_factor(cfg, gc_soft)

    # State mixture
    states = [{"name": s.name, "weight": s.weight, "price_multiplier": s.price_multiplier, "delay_mean_add": s.delay_mean_add, "delay_var_mult": s.delay_var_mult} for s in cfg.qsi.states]
    if not states:
        states = [{"name": "base", "weight": 1.0, "price_multiplier": 1.0, "delay_mean_add": 0.0, "delay_var_mult": 1.0}]
    # Normalize weights
    sw = sum(s["weight"] for s in states)
    for s in states:
        s["weight"] = s["weight"] / max(sw, 1e-6)

    # Compute itinerary weights per segment and state
    W_sum = 0.0
    for s in segments:
        w_s = next((seg.weight for seg in cfg.qsi.segments if seg.name == s), 1.0)
        for st in states:
            for k in itineraries:
                W_sum += w_s * st["weight"] * itinerary_weight(cfg, k, s, st)

    qsi_plus = A * W_sum
    return {"A": A, "W_sum": W_sum, "QSI_plus": qsi_plus}


