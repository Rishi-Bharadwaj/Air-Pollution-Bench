#!/usr/bin/env python3
"""
Compute Overall Leaderboard from local TIME evaluation results.

This script:
1. Loads all model results (including seasonal_naive) from output/results/
2. Computes Overall leaderboard metrics (normalized by Seasonal Naive)
3. Computes per-pollutant leaderboard (if item_ids available)
4. Exports results to CSV

Usage:
    python scripts/compute_local_leaderboard.py
    python scripts/compute_local_leaderboard.py --dataset CPCB/H --metric MASE

Requirements:
    - pandas
    - numpy
    - scipy
    - pyyaml
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory to path to import timebench utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from leaderboard_utils import extract_pollutant, display_dataset, to_latex_table, MODEL_GROUPS, GROUP_ORDER
from leaderboard_helpers import (
    normalize_by_seasonal_naive,
    check_result_consistency,
    SEASONAL_NAIVE_MODEL,
)


def _iter_model_series(results_root: Path, dataset_filter: list[str] = None):
    """Iterate over (model, dataset_id, horizon, item_ids, npz_metrics) tuples."""
    import json
    for model_dir in results_root.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            for freq_dir in dataset_dir.iterdir():
                if not freq_dir.is_dir():
                    continue
                freq_name = freq_dir.name
                dataset_id = f"{dataset_name}/{freq_name}"
                if dataset_filter and dataset_id not in dataset_filter:
                    continue
                for horizon in ["short", "medium", "long"]:
                    horizon_dir = results_root / model_name / dataset_id / horizon
                    metrics_path = horizon_dir / "metrics.npz"
                    config_path = horizon_dir / "config.json"
                    if not metrics_path.exists() or not config_path.exists():
                        continue
                    with open(config_path) as f:
                        config = json.load(f)
                    item_ids = config.get("item_ids")
                    if not item_ids:
                        continue
                    npz_metrics = np.load(metrics_path)
                    yield model_name, dataset_id, horizon, item_ids, npz_metrics


def get_per_pollutant_results(results_root: Path, dataset_filter: list[str] = None) -> pd.DataFrame:
    """
    Load per-series metrics from NPZ files, map to pollutant via item_ids in config.json,
    and return per-pollutant aggregated metrics.

    Sites where MASE > threshold for the mean of models are excluded
    to ensure a fair comparison.

    Returns:
        DataFrame with columns ["model", "dataset_id", "horizon", "pollutant", "MASE", "CRPS", "MAE", "RMSE"]
    """
    THRESHOLD = 50

    # --- Pass 1: collect per-site MASE and CRPS across all models, then exclude by mean ---
    # Key: (dataset_id, horizon, item_id) -> list of values across models
    site_metric_values: dict[str, dict[tuple[str, str, str], list[float]]] = {"MASE": {}, "CRPS": {}}

    for model_name, dataset_id, horizon, item_ids, npz_metrics in _iter_model_series(results_root, dataset_filter):
        n_series = len(item_ids)
        for metric_name in ["MASE", "CRPS"]:
            arr = npz_metrics.get(metric_name)
            if arr is None or arr.shape[0] != n_series:
                continue
            reduce_axes = tuple(range(1, arr.ndim))
            per_series = np.nanmean(arr[:n_series], axis=reduce_axes) if reduce_axes else arr[:n_series]
            for i, iid in enumerate(item_ids):
                val = per_series[i]
                if not np.isnan(val):
                    site_metric_values[metric_name].setdefault((dataset_id, horizon, iid), []).append(float(val))

    # --- Diagnostic: check which (dataset_id, horizon) configs each model has ---
    model_configs: dict[str, set[tuple[str, str]]] = {}
    for model_name, dataset_id, horizon, item_ids, _ in _iter_model_series(results_root, dataset_filter):
        model_configs.setdefault(model_name, set()).add((dataset_id, horizon))

    all_configs = set().union(*model_configs.values())
    print(f"\n  Dataset/horizon configs seen: {len(all_configs)} total")
    for model_name, configs in sorted(model_configs.items()):
        print(f"    {model_name}: {len(configs)} configs")
        missing = all_configs - configs
        if missing:
            print(f"      ⚠️  missing: {sorted(missing)}")

    # Exclude sites where mean MASE > threshold OR mean CRPS > threshold
    excluded_sites: dict[tuple[str, str], set[str]] = {}
    all_site_keys = set(site_metric_values["MASE"]) | set(site_metric_values["CRPS"])
    for key in all_site_keys:
        dataset_id, horizon, iid = key
        mase_vals = site_metric_values["MASE"].get(key, [])
        crps_vals = site_metric_values["CRPS"].get(key, [])
        if (mase_vals and np.mean(mase_vals) > THRESHOLD) or (crps_vals and np.mean(crps_vals) > THRESHOLD):
            excluded_sites.setdefault((dataset_id, horizon), set()).add(iid)

    # Log excluded sites with pollutant info
    if excluded_sites:
        print(f"\n  Threshold ({THRESHOLD}) exclusions — mean MASE>threshold OR mean CRPS>threshold (applied to ALL models):")
        for (ds, hz), ids in sorted(excluded_sites.items()):
            pollutant_counts: dict[str, int] = {}
            for iid in ids:
                pol = extract_pollutant(iid)
                pollutant_counts[pol] = pollutant_counts.get(pol, 0) + 1
            breakdown = ", ".join(f"{pol}: {n}" for pol, n in sorted(pollutant_counts.items()))
            print(f"    {ds}/{hz}: {len(ids)} site(s) excluded ({breakdown})")

    # --- Pass 2: load all metrics, masking excluded sites by item_id ---
    rows = []
    for model_name, dataset_id, horizon, item_ids, npz_metrics in _iter_model_series(results_root, dataset_filter):
        n_series = len(item_ids)
        key = (dataset_id, horizon)
        exclude_ids = excluded_sites.get(key, set())

        batch = {
            "model": [model_name] * n_series,
            "dataset_id": [dataset_id] * n_series,
            "horizon": [horizon] * n_series,
            "pollutant": [extract_pollutant(iid) for iid in item_ids],
        }
        for metric_name in ["MASE", "CRPS", "MAE", "RMSE"]:
            arr = npz_metrics.get(metric_name)
            if arr is not None and arr.shape[0] == n_series:
                reduce_axes = tuple(range(1, arr.ndim))
                if metric_name == "RMSE":
                    per_series = np.sqrt(np.nanmean(arr[:n_series] ** 2, axis=reduce_axes)).copy() if reduce_axes else arr[:n_series]
                else:
                    per_series = np.nanmean(arr[:n_series], axis=reduce_axes).copy() if reduce_axes else arr[:n_series]
                # Mask excluded sites for MASE and CRPS
                if metric_name in ("MASE", "CRPS"):
                    for i, iid in enumerate(item_ids):
                        if iid in exclude_ids:
                            per_series[i] = np.nan
                batch[metric_name] = per_series.tolist()
            else:
                batch[metric_name] = [np.nan] * n_series
        rows.append(pd.DataFrame(batch))

    if not rows:
        return pd.DataFrame(columns=["model", "dataset_id", "horizon", "pollutant", "MASE", "CRPS", "MAE", "RMSE"])

    # Aggregate per (model, dataset_id, horizon, pollutant)
    df = pd.concat(rows, ignore_index=True)
    return df.groupby(["model", "dataset_id", "horizon", "pollutant"], as_index=False)[
        ["MASE", "CRPS", "MAE", "RMSE"]
    ].mean()


def _save_per_dataset_horizon_tables(
    balanced_norm: pd.DataFrame,
    output_dir: Path,
    metric: str,
    model_groups: dict | None = None,
    group_order: list | None = None,
) -> None:
    """Save per-(dataset_id, horizon) normalized leaderboard LaTeX tables and CSVs."""
    subdir = output_dir / "per_dataset_horizon"
    csv_subdir = output_dir / "per_dataset_horizon_csv"
    subdir.mkdir(parents=True, exist_ok=True)
    csv_subdir.mkdir(parents=True, exist_ok=True)

    df = balanced_norm.rename(columns={"MASE": "MASE (norm.)", "CRPS": "CRPS (norm.)"})
    sort_col = "MASE (norm.)" if metric == "MASE" else "CRPS (norm.)"

    table_num = 1
    for (dataset_id, horizon), grp_df in df.groupby(["dataset_id", "horizon"]):
        tbl = grp_df[["model", "MASE (norm.)", "CRPS (norm.)"]].copy().round(3)
        if model_groups is None:
            tbl = tbl.sort_values(by=sort_col, ascending=True).reset_index(drop=True)
        caption = f"Normalized leaderboard --- {display_dataset(dataset_id)} --- {horizon}"
        tex = to_latex_table(
            tbl, caption, table_num,
            metric_cols=["MASE (norm.)", "CRPS (norm.)"],
            model_groups=model_groups,
            group_order=group_order,
        )
        stem = f"{dataset_id.replace('/', '_')}_{horizon}"
        (subdir / f"{stem}.tex").write_text(tex)
        tbl.sort_values(by=sort_col, ascending=True).reset_index(drop=True).to_csv(
            csv_subdir / f"{stem}.csv", index=False
        )
        table_num += 1
    print(f"   Saved {table_num - 1} per-(dataset, horizon) tables to {subdir} and {csv_subdir}")


def get_pollutant_balanced_leaderboard(
    pollutant_results: pd.DataFrame,
    metric: str = "MASE",
    output_dir: Path | None = None,
    model_groups: dict | None = None,
    group_order: list | None = None,
) -> pd.DataFrame:
    """
    Compute a pollutant-balanced overall leaderboard.

    1. Mean MASE/CRPS per (model, dataset, horizon, pollutant) — across sites
    2. Mean of those per (model, dataset, horizon) — balanced per-dataset score
    3. Normalize by Seasonal Naive's balanced score per (dataset, horizon)
       → if output_dir given, saves per-(dataset, horizon) LaTeX tables here
    4. Geometric mean across (dataset, horizon) configs

    Returns leaderboard DataFrame similar to get_overall_leaderboard.
    """
    if pollutant_results.empty:
        return pd.DataFrame()

    # Step 1: mean per (model, dataset, horizon, pollutant)
    per_pol = pollutant_results.groupby(
        ["model", "dataset_id", "horizon", "pollutant"], as_index=False
    )[["MASE", "CRPS"]].mean()

    # Step 2: mean across pollutants per (model, dataset, horizon)
    balanced = per_pol.groupby(
        ["model", "dataset_id", "horizon"], as_index=False
    )[["MASE", "CRPS"]].mean()

    # Step 3: normalize by seasonal naive
    balanced_norm = normalize_by_seasonal_naive(
        balanced,
        baseline_model=SEASONAL_NAIVE_MODEL,
        metrics=["MASE", "CRPS"],
        groupby_cols=["dataset_id", "horizon"],
    )

    if balanced_norm.empty:
        return pd.DataFrame()

    # Save per-(dataset, horizon) tables before aggregating across datasets
    if output_dir is not None:
        _save_per_dataset_horizon_tables(balanced_norm, output_dir, metric, model_groups, group_order)

    # Step 4: geometric mean across (dataset, horizon) configs
    def gmean_with_nan(x):
        valid = x.dropna()
        if len(valid) == 0:
            return np.nan
        return stats.gmean(valid)

    leaderboard = (
        balanced_norm.groupby("model")[["MASE", "CRPS"]]
        .agg(gmean_with_nan)
        .reset_index()
    )
    leaderboard = leaderboard.rename(columns={
        "MASE": "MASE (norm.)",
        "CRPS": "CRPS (norm.)",
    })

    sort_col = "MASE (norm.)" if metric == "MASE" else "CRPS (norm.)"
    if sort_col in leaderboard.columns:
        leaderboard = leaderboard.sort_values(by=sort_col, ascending=True).reset_index(drop=True)

    leaderboard = leaderboard.round(3)
    return leaderboard



def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Compute leaderboard from TIME evaluation results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory (default: output/results)")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="Dataset(s) to include, e.g. CPCB/H (default: all)")
    parser.add_argument("--metric", type=str, default=None, choices=["MASE", "CRPS"],
                        help="Metric to sort by (default: MASE)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save CSV exports (default: output/leaderboard)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (used for defaults if CLI args not given)")
    args = parser.parse_args()

    # Load defaults from config.yaml leaderboard section
    config_lb = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        config_lb = config.get("leaderboard", {}) or {}

    results_dir = args.results_dir or config_lb.get("results_dir", "output/results")
    dataset_filter = args.dataset or config_lb.get("datasets", None)
    metric = args.metric or config_lb.get("metric", "MASE")
    output_dir = Path(args.output_dir or config_lb.get("output_dir", "output/leaderboard"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AirBench Local Leaderboard Calculator")
    print("=" * 80)
    if dataset_filter:
        print(f"  Filtering to datasets: {dataset_filter}")
    print()

    results_root = Path(results_dir)

    if not results_root.exists():
        print(f"❌ Error: Results directory does not exist: {results_root}")
        sys.exit(1)

    if not (results_root / SEASONAL_NAIVE_MODEL).is_dir():
        print(f"❌ No '{SEASONAL_NAIVE_MODEL}' results found in {results_root}.")
        print(f"   Run seasonal_naive first, or place its results in {results_root / SEASONAL_NAIVE_MODEL}/")
        sys.exit(1)

    # Step 1: Consistency check
    print(f"Step 1: Checking result consistency across models and datasets...")
    check_result_consistency(results_root, dataset_filter)

    # Step 2: Per-pollutant results
    print(f"\nStep 2: Computing per-pollutant leaderboard...")
    pollutant_results = get_per_pollutant_results(results_root, dataset_filter)

    if pollutant_results.empty:
        print("   No per-pollutant data available (item_ids missing from config.json?)")
        sys.exit(1)

    pollutants = sorted(pollutant_results["pollutant"].unique())
    print(f"   Found pollutants: {pollutants}")

    # Build per-pollutant tables: mean across sites per pollutant
    pol_subdir = output_dir / "per_pollutant"
    pol_csv_subdir = output_dir / "per_pollutant_csv"
    pol_subdir.mkdir(parents=True, exist_ok=True)
    pol_csv_subdir.mkdir(parents=True, exist_ok=True)
    table_num = 1
    datasets_in_results = sorted(pollutant_results["dataset_id"].unique())
    for dataset_id in datasets_in_results:
        ddf = pollutant_results[pollutant_results["dataset_id"] == dataset_id]
        dataset_pollutants = sorted(ddf["pollutant"].unique())

        # Subfolder per dataset, using only the dataset name (no freq suffix)
        dataset_subdir = pol_subdir / dataset_id.split("/")[0]
        dataset_csv_subdir = pol_csv_subdir / dataset_id.split("/")[0]
        dataset_subdir.mkdir(parents=True, exist_ok=True)
        dataset_csv_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  Dataset: {dataset_id}")
        print(f"{'=' * 60}")

        for pollutant in dataset_pollutants:
            pdf = ddf[ddf["pollutant"] == pollutant]
            agg = pdf.groupby("model")[["MASE", "CRPS", "MAE", "RMSE"]].mean().reset_index()
            agg = agg.sort_values(by=metric, ascending=True).reset_index(drop=True)
            agg = agg.round(3)

            print(f"\n  {'─' * 40}")
            print(f"    Pollutant: {pollutant}")
            print(f"  {'─' * 40}")
            print(agg.to_string(index=False))


            # Save individual LaTeX table and CSV
            caption = f"{pollutant} leaderboard --- {display_dataset(dataset_id)}"
            tex = to_latex_table(agg, caption, table_num, metric_cols=["MASE", "CRPS", "MAE", "RMSE"],
                                 model_groups=MODEL_GROUPS, group_order=GROUP_ORDER)
            (dataset_subdir / f"{pollutant}.tex").write_text(tex)
            agg.to_csv(dataset_csv_subdir / f"{pollutant}.csv", index=False)
            table_num += 1

    print()

    # Pollutant-balanced overall leaderboard
    balanced_lb = get_pollutant_balanced_leaderboard(
        pollutant_results, metric=metric,
        output_dir=output_dir,
        model_groups=MODEL_GROUPS,
        group_order=GROUP_ORDER,
    )
    if not balanced_lb.empty:
        print(f"\n{'=' * 60}")
        print("  Pollutant-Balanced Overall Leaderboard")
        print("  (mean across sites per pollutant, mean across pollutants per dataset,")
        print("   normalized by Seasonal Naive, gmean across datasets)")
        print(f"{'=' * 60}")
        print(balanced_lb.to_string(index=False))
        print()

        balanced_csv = output_dir / "pollutant_balanced_leaderboard.csv"
        balanced_lb.to_csv(balanced_csv, index=False)
        print(f"   Saved pollutant-balanced leaderboard to {balanced_csv}")

        balanced_tex = output_dir / "pollutant_balanced_leaderboard.tex"
        balanced_caption = "Pollutant-balanced overall leaderboard (normalized by Seasonal Naive, gmean across datasets)"
        balanced_tex.write_text(to_latex_table(
            balanced_lb, balanced_caption, 2,
            metric_cols=["MASE (norm.)", "CRPS (norm.)"],
            model_groups=MODEL_GROUPS, group_order=GROUP_ORDER,
        ))
        print(f"   Saved pollutant-balanced LaTeX table to {balanced_tex}")

if __name__ == "__main__":
    main()