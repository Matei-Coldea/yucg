from __future__ import annotations

import argparse
import os

import pandas as pd

from .config import load_config
from .data import load_ct_zctas, ct_zip_centroids
from .model import generalized_cost_delta
from .plotting import choropleth_delta


def run(config_path: str, outdir: str) -> None:
    cfg = load_config(config_path)
    zctas = load_ct_zctas()
    origins = ct_zip_centroids(zctas)
    res = generalized_cost_delta(cfg, origins, cfg.region.airports, cfg.region.modes)
    os.makedirs(outdir, exist_ok=True)
    res.to_csv(os.path.join(outdir, "gcdm_results.csv"), index=False)
    choropleth_delta(zctas, res, cfg.map.title, outdir, crs=cfg.map.output_crs, interactive=cfg.map.interactive)


def main():
    parser = argparse.ArgumentParser(description="Run GCDM for Connecticut")
    parser.add_argument("run", nargs="?")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    run(args.config, args.outdir)


if __name__ == "__main__":
    main()


