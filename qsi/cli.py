from __future__ import annotations

import argparse
import os
import pandas as pd

from gcdm.config import load_config
from gcdm.data import load_ct_zctas, ct_zip_centroids
from gcdm.model import generalized_cost_delta
from gcdm.qsi import qsi_plus_for_market, Itinerary


def run_qsi(config_path: str, outdir: str, itins_csv: str | None = None) -> None:
    cfg = load_config(config_path)
    zctas = load_ct_zctas()
    origins = ct_zip_centroids(zctas)
    mid = origins.iloc[len(origins) // 2 : len(origins) // 2 + 1]
    res = generalized_cost_delta(cfg, mid, cfg.region.airports, cfg.region.modes)
    per_mode_gc = {m: 30.0 for m in cfg.region.modes}

    itins: list[Itinerary] = []
    if itins_csv:
        df = pd.read_csv(itins_csv)
        for _, r in df.iterrows():
            itins.append(Itinerary(**r.to_dict()))
    else:
        itins.append(Itinerary(
            origin_airport="HVN", hub="CLT", dest="DFW", mode_access="drive", block_minutes=240, layover_minutes=60, L_star=45, U_star=120, sk=0.2,
            tdep_iso="2025-09-13T07:00:00-04:00", tarr_iso="2025-09-13T13:40:00-05:00", cancel_rate=0.03, inbound_delay_mean=10, inbound_taxi_in_mean=8, delay_pos_mean=12, delay_var=25,
            hub_WxRisk=0.2, hub_CapacityRisk=0.3, reprotect_time_mean=180, n_alternates_window=3, seat_pitch=31, wifi_rel=0.9, is_widebody=0, is_priority=0, has_lounge=0, has_precheck=1,
            price_mean=280, price_ancillary=25, price_rebates=10, seats_available=5, dupcount=1, loadfactor_risk=0.2, bag_miss_prob=0.05, hub_bank_times=["08:00","12:00","17:00"]
        ))
        itins.append(Itinerary(
            origin_airport="HVN", hub="ATL", dest="LAX", mode_access="rail", block_minutes=320, layover_minutes=75, L_star=45, U_star=120, sk=0.3,
            tdep_iso="2025-09-13T08:10:00-04:00", tarr_iso="2025-09-13T14:50:00-07:00", cancel_rate=0.04, inbound_delay_mean=12, inbound_taxi_in_mean=9, delay_pos_mean=15, delay_var=40,
            hub_WxRisk=0.25, hub_CapacityRisk=0.35, reprotect_time_mean=220, n_alternates_window=4, seat_pitch=32, wifi_rel=0.8, is_widebody=1, is_priority=1, has_lounge=1, has_precheck=1,
            price_mean=420, price_ancillary=35, price_rebates=20, seats_available=3, dupcount=0, loadfactor_risk=0.3, bag_miss_prob=0.07, hub_bank_times=["09:00","13:00","18:00"]
        ))

    segs = [s.name for s in cfg.qsi.segments] or ["domestic", "intl"]
    out = qsi_plus_for_market(cfg, per_mode_gc, itins, segs)
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame([out]).to_csv(os.path.join(outdir, "qsi_plus_summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Run QSI+ engine")
    parser.add_argument("run", nargs="?")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--itins_csv")
    args = parser.parse_args()
    run_qsi(args.config, args.outdir, args.itins_csv)


if __name__ == "__main__":
    main()


