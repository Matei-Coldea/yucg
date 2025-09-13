from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from .config import AppConfig


def cvar(samples: np.ndarray, alpha: float) -> float:
    if samples.size == 0:
        return 0.0
    q = np.quantile(samples, alpha)
    tail = samples[samples >= q]
    if tail.size == 0:
        return float(q)
    return float(tail.mean())


def cvar_minus_mean(samples: np.ndarray, alpha: float) -> float:
    if samples.size == 0:
        return 0.0
    return cvar(samples, alpha) - float(samples.mean())


def gaussian_samples(mean: float, sd: float, size: int = 2000, lower: float = 0.0) -> np.ndarray:
    x = np.random.normal(mean, sd, size)
    if lower is not None:
        x = np.clip(x, lower, None)
    return x


def apply_context_multiplier(x: float, k_weather: float, k_event: float, k_construction: float) -> float:
    return x * k_weather * k_event * k_construction


def apply_shocks(samples: np.ndarray, shock_prob: float, shock_minutes: float) -> np.ndarray:
    if shock_prob <= 0 or shock_minutes <= 0:
        return samples
    mask = np.random.rand(samples.shape[0]) < shock_prob
    samples = samples.copy()
    samples[mask] += shock_minutes
    return samples


def risk_with_shocks(cfg: AppConfig, A_samples: np.ndarray, P_samples: np.ndarray) -> float:
    alpha = cfg.risk.alpha
    rho = cfg.risk.rho
    # Apply shock models
    A_shocked = apply_shocks(A_samples, cfg.risk.shocks.get("incident_prob", 0.0), cfg.risk.shocks.get("incident_minutes", 0.0))
    P_shocked = apply_shocks(P_samples, cfg.risk.shocks.get("rail_disruption_prob", 0.0), cfg.risk.shocks.get("rail_disruption_minutes", 0.0))
    deltaA = cvar_minus_mean(A_shocked, alpha)
    deltaP = cvar_minus_mean(P_shocked, alpha)
    return float(rho * (deltaA + deltaP))


