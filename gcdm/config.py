from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


@dataclass
class ProjectConfig:
    name: str
    seed: int
    processes: int


@dataclass
class RegionConfig:
    focus_state_fips: str
    airports: List[str]
    modes: List[str]


@dataclass
class PartyConfig:
    NA: int
    NC: int
    vA_usd_per_hour: float
    vC_usd_per_hour: float
    lambda_child_time_weight: float


@dataclass
class ScheduleWindow:
    tmin: str
    tmax: str


@dataclass
class ScheduleConfig:
    t_depart_iso: str
    curb_windows: Dict[str, ScheduleWindow]


@dataclass
class VOTBand:
    name: str
    minutes_value: float


@dataclass
class VOTConfig:
    tod_bands: List[VOTBand]
    peak_hours: List[int]


@dataclass
class RiskMultipliers:
    weather: float
    event: float
    construction: float


@dataclass
class RiskConfig:
    alpha: float
    rho: float
    multipliers: RiskMultipliers


@dataclass
class DriveCostConfig:
    cmile: float
    tolled_miles_multiplier: float
    rem_lot_fee: float
    shuttle_fee: float
    fare_risk: float


@dataclass
class DriveModeConfig:
    cost: DriveCostConfig
    parking_daily: Dict[str, float]
    parking_weekly: Dict[str, float]


@dataclass
class RideModeConfig:
    surge: float
    base_fare: float
    per_mile: float
    per_minute: float
    booking_fee: float
    tip_fraction: float
    fare_risk: float


@dataclass
class RailModeConfig:
    adult_fare_default: float
    child_fare_default: float
    transfer_fare: float
    airtrain: Dict[str, float]
    station_parking: float
    miss_prob_base: float
    headway_minutes: int


@dataclass
class ModesConfig:
    drive: DriveModeConfig
    ride: RideModeConfig
    rail: RailModeConfig


@dataclass
class ProcessTimesConfig:
    security_mean: Dict[str, float]
    bag_drop_mean: Dict[str, float]
    walk_mean: Dict[str, float]


@dataclass
class ComfortConfig:
    omega_peak: float
    omega_night: float


@dataclass
class CarbonConfig:
    enabled: bool
    social_cost_per_ton: float
    emissions_kg_per_mile: Dict[str, float]


@dataclass
class SoftMinConfig:
    mu: float


@dataclass
class MapConfig:
    output_crs: int
    interactive: bool
    basemap: str
    title: str


@dataclass
class ProvidersConfig:
    enable_google: bool
    enable_openrouteservice: bool
    enable_weather: bool
    enable_events: bool
    google_api_key: str
    ors_api_key: str
    owm_api_key: str
    ticketmaster_api_key: str


@dataclass
class AppConfig:
    project: ProjectConfig
    region: RegionConfig
    party: PartyConfig
    schedule: ScheduleConfig
    vot: VOTConfig
    risk: RiskConfig
    modes: ModesConfig
    process_times: ProcessTimesConfig
    comfort: ComfortConfig
    carbon: CarbonConfig
    softmin: SoftMinConfig
    map: MapConfig
    providers: ProvidersConfig
    qsi: "QSIConfig"


def load_config(path: str) -> AppConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Helper for nested dataclasses
    def dc(klass, data):
        return klass(**data)

    project = dc(ProjectConfig, raw["project"])
    region = dc(RegionConfig, raw["region"])
    party = dc(PartyConfig, raw["party"])

    # schedule windows
    curb_windows = {
        k: dc(ScheduleWindow, v) for k, v in raw["schedule"]["curb_windows"].items()
    }
    schedule = ScheduleConfig(
        t_depart_iso=raw["schedule"]["t_depart_iso"],
        curb_windows=curb_windows,
    )

    vot = VOTConfig(
        tod_bands=[dc(VOTBand, x) for x in raw["vot"]["tod_bands"]],
        peak_hours=list(raw["vot"]["peak_hours"]),
    )

    risk = RiskConfig(
        alpha=float(raw["risk"]["alpha"]),
        rho=float(raw["risk"]["rho"]),
        multipliers=dc(RiskMultipliers, raw["risk"]["multipliers"]),
    )

    modes = ModesConfig(
        drive=DriveModeConfig(
            cost=dc(DriveCostConfig, raw["modes"]["drive"]["cost"]),
            parking_daily=dict(raw["modes"]["drive"]["parking_daily"]),
            parking_weekly=dict(raw["modes"]["drive"]["parking_weekly"]),
        ),
        ride=dc(RideModeConfig, raw["modes"]["ride"]),
        rail=dc(RailModeConfig, raw["modes"]["rail"]),
    )

    process_times = dc(ProcessTimesConfig, raw["process_times"])
    comfort = dc(ComfortConfig, raw["comfort"])
    carbon = dc(CarbonConfig, raw["carbon"])
    softmin = dc(SoftMinConfig, raw["softmin"])
    map_cfg = dc(MapConfig, raw["map"])
    providers = dc(ProvidersConfig, raw.get("providers", {}))

    return AppConfig(
        project=project,
        region=region,
        party=party,
        schedule=schedule,
        vot=vot,
        risk=risk,
        modes=modes,
        process_times=process_times,
        comfort=comfort,
        carbon=carbon,
        softmin=softmin,
        map=map_cfg,
        providers=providers,
        qsi=parse_qsi(raw.get("qsi", {})),
    )


# ===================== QSI+ CONFIG =====================
from dataclasses import field


@dataclass
class QSIState:
    name: str
    weight: float
    price_multiplier: float
    delay_mean_add: float
    delay_var_mult: float


@dataclass
class QSISegment:
    name: str
    weight: float
    beta_P: float
    p_star: float
    seats_threshold: int


@dataclass
class QSIQualityWeights:
    wseat: float
    wwifi: float
    wac: float
    wprio: float


@dataclass
class QSIDestWeight:
    code: str
    weight: float


@dataclass
class QSIHubRisk:
    WxRisk: float
    CapacityRisk: float
    MCT: float
    bank_times: list


@dataclass
class QSIConfig:
    muA: float
    betaA: float
    betaD: float
    betaV: float
    betaW: float
    betaC: float
    beta_rec: float
    xi: float
    betaQ: float
    quality_weights: QSIQualityWeights
    beta_ffp: float
    beta_lounge: float
    beta_pre: float
    beta_scar: float
    beta_B: float
    bank_window_w: float
    beta_MCT: float
    beta_bag: float
    alt_window_minutes: float
    share_theta: float
    states: list
    segments: list
    dests: list
    hub_risks: dict


def parse_qsi(raw: dict) -> QSIConfig:
    def dc(klass, data):
        return klass(**data)

    quality = dc(QSIQualityWeights, raw.get("quality_weights", {"wseat": 0.5, "wwifi": 0.2, "wac": 0.2, "wprio": 0.1}))
    states = [dc(QSIState, s) for s in raw.get("states", [])]
    segments = [dc(QSISegment, s) for s in raw.get("segments", [])]
    dests = [dc(QSIDestWeight, d) for d in raw.get("dests", [])]
    hub_risks = raw.get("hub_risks", {})

    return QSIConfig(
        muA=float(raw.get("muA", 0.2)),
        betaA=float(raw.get("betaA", 0.03)),
        betaD=float(raw.get("betaD", 0.02)),
        betaV=float(raw.get("betaV", 0.01)),
        betaW=float(raw.get("betaW", 0.02)),
        betaC=float(raw.get("betaC", 0.02)),
        beta_rec=float(raw.get("beta_rec", 0.02)),
        xi=float(raw.get("xi", 0.15)),
        betaQ=float(raw.get("betaQ", 0.02)),
        quality_weights=quality,
        beta_ffp=float(raw.get("beta_ffp", 0.01)),
        beta_lounge=float(raw.get("beta_lounge", 0.01)),
        beta_pre=float(raw.get("beta_pre", 0.01)),
        beta_scar=float(raw.get("beta_scar", 0.02)),
        beta_B=float(raw.get("beta_B", 0.02)),
        bank_window_w=float(raw.get("bank_window_w", 1.0)),
        beta_MCT=float(raw.get("beta_MCT", 0.03)),
        beta_bag=float(raw.get("beta_bag", 0.02)),
        alt_window_minutes=float(raw.get("alt_window_minutes", 180.0)),
        share_theta=float(raw.get("share_theta", 1.0)),
        states=states,
        segments=segments,
        dests=dests,
        hub_risks=hub_risks,
    )


