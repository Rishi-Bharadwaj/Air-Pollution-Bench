"""Build HF Arrow datasets for air quality data from aq_datasets.yaml config."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import yaml
from timebench.evaluation.dataset_builder import build_dataset_from_csvs


def main():
    parser = argparse.ArgumentParser(description="Build AQ HF datasets from config")
    parser.add_argument(
        "--config", type=str, default="aq_datasets.yaml", help="Path to aq_datasets.yaml"
    )
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=None,
        help="Specific dataset(s) to build (default: all)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    datasets = config["sources"]
    if args.dataset:
        datasets = {k: v for k, v in datasets.items() if k in args.dataset}

    for name, cfg in datasets.items():
        print(f"\n{'='*60}")
        print(f"Building: {name}")
        print(f"  Input:  {cfg['csv_dir']}")
        print(f"  Output: {cfg['output_path']}")
        print(f"{'='*60}")

        ds = build_dataset_from_csvs(
            csv_dir=cfg["csv_dir"], output_path=cfg["output_path"], to_univariate=True
        )
        print(f"  Series: {len(ds)}")
        print(f"  Features: {ds.column_names}")

    print(f"\nDone. Built {len(datasets)} dataset(s).")


if __name__ == "__main__":
    main()