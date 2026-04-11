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

SEASONAL_NAIVE_MODEL = "seasonal_naive"


def load_time_results(root_dir: Path, model_name: str, dataset_with_freq: str, horizon: str):
    """
    Load TIME results from NPZ files for a specific model, dataset, and horizon.

    Args:
        root_dir: Root directory containing TIME results
        model_name: Model name (e.g., "chronos2")
        dataset_with_freq: Dataset and freq combined (e.g., "Traffic/15T")
        horizon: Horizon name (e.g., "short", "medium", "long")

    Returns:
        tuple: (metrics_dict, config_dict) or (None, None) if not found
    """
    horizon_dir = root_dir / model_name / dataset_with_freq / horizon
    metrics_path = horizon_dir / "metrics.npz"
    config_path = horizon_dir / "config.json"

    if not metrics_path.exists():
        return None, None

    metrics = np.load(metrics_path)
    metrics_dict = {k: metrics[k] for k in metrics.files}

    config_dict = {}
    if config_path.exists():
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)

    return metrics_dict, config_dict


def get_all_datasets_results(results_root: Path) -> pd.DataFrame:
    """
    Load dataset-level leaderboard by reading TIME NPZ files and aggregating.

    Args:
        results_root: Path to the TIME results root directory

    Returns:
        pd.DataFrame: DataFrame containing dataset-level results with columns
            ["model", "dataset", "freq", "dataset_id", "horizon", "MASE", "CRPS", "MAE", "RMSE"]
    """
    rows = []

    if not results_root.exists():
        print(f"❌ Error: results_root={results_root} does not exist")
        return pd.DataFrame(columns=["model", "dataset", "freq", "dataset_id", "horizon", "MASE", "CRPS", "MAE", "RMSE"])

    for model_dir in results_root.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            # Nested structure: model/dataset/freq/horizon/
            for freq_dir in dataset_dir.iterdir():
                if not freq_dir.is_dir():
                    continue

                freq_name = freq_dir.name

                for horizon in ["short", "medium", "long"]:
                    dataset_with_freq = f"{dataset_name}/{freq_name}"
                    metrics_dict, _ = load_time_results(results_root, model_name, dataset_with_freq, horizon)

                    if metrics_dict is None:
                        continue

                    # Aggregate metrics across all series/windows/variates
                    mase = np.nanmean(metrics_dict.get("MASE", np.array([])))
                    crps = np.nanmean(metrics_dict.get("CRPS", np.array([])))
                    mae = np.nanmean(metrics_dict.get("MAE", np.array([])))
                    rmse = np.nanmean(metrics_dict.get("RMSE", np.array([])))

                    rows.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "freq": freq_name,
                        "dataset_id": dataset_with_freq,
                        "horizon": horizon,
                        "MASE": mase,
                        "CRPS": crps,
                        "MAE": mae,
                        "RMSE": rmse,
                    })

    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(columns=["model", "dataset", "freq", "dataset_id", "horizon", "MASE", "CRPS", "MAE", "RMSE"])


def compute_ranks(df: pd.DataFrame, groupby_cols: list) -> pd.DataFrame:
    """
    Compute ranks for models across datasets based on MASE and CRPS.

    Args:
        df: Dataset-level results with columns ["model", "dataset_id", "horizon", "MASE", "CRPS"]
        groupby_cols: Columns to group by for ranking

    Returns:
        DataFrame with added ["MASE_rank", "CRPS_rank"] columns
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df["MASE_rank"] = df.groupby(groupby_cols)["MASE"].rank(method="first", ascending=True)
    df["CRPS_rank"] = df.groupby(groupby_cols)["CRPS"].rank(method="first", ascending=True)

    return df


def normalize_by_seasonal_naive(
    df: pd.DataFrame,
    baseline_model: str = "seasonal_naive",
    metrics: list = None,
    groupby_cols: list = None,
) -> pd.DataFrame:
    """
    Normalize metrics by Seasonal Naive baseline for each (dataset_id, horizon) group.

    Args:
        df: Dataset-level results with columns including ["model", "dataset_id", "horizon", "MASE", "CRPS"]
        baseline_model: Name of the baseline model
        metrics: List of metric columns to normalize
        groupby_cols: Columns to group by for normalization

    Returns:
        DataFrame with normalized metric values
    """
    if metrics is None:
        metrics = ["MASE", "CRPS"]
    if groupby_cols is None:
        groupby_cols = ["dataset_id", "horizon"]

    if df.empty:
        return df.copy()

    # Check if baseline model exists
    if baseline_model not in df["model"].values:
        print(f"⚠️  Warning: baseline model '{baseline_model}' not found in data")
        return pd.DataFrame()

    # Work on a copy
    df_normalized = df.copy()

    # Get baseline values for each group
    baseline_df = df[df["model"] == baseline_model].copy()

    # Create a mapping: (dataset_id, horizon) -> {metric: baseline_value}
    baseline_values = {}
    for _, row in baseline_df.iterrows():
        key = tuple(row[col] for col in groupby_cols)
        baseline_values[key] = {metric: row[metric] for metric in metrics}

    # Normalize each row
    rows_to_keep = []
    for idx, row in df_normalized.iterrows():
        key = tuple(row[col] for col in groupby_cols)

        # Skip configurations without baseline results
        if key not in baseline_values:
            continue

        rows_to_keep.append(idx)

        # Normalize each metric
        for metric in metrics:
            baseline_val = baseline_values[key][metric]
            if baseline_val is not None and baseline_val != 0 and not np.isnan(baseline_val):
                df_normalized.at[idx, metric] = row[metric] / baseline_val
            else:
                df_normalized.at[idx, metric] = np.nan

    # Keep only rows with valid baseline
    df_normalized = df_normalized.loc[rows_to_keep].copy()

    # Handle any remaining inf values
    for metric in metrics:
        df_normalized[metric] = df_normalized[metric].replace([np.inf, -np.inf], np.nan)

    return df_normalized


def get_overall_leaderboard(df_datasets: pd.DataFrame, metric: str = "MASE") -> pd.DataFrame:
    """
    Compute overall leaderboard across datasets by normalizing metrics by Seasonal Naive
    and aggregating with geometric mean.

    Args:
        df_datasets: Dataset-level results, must include
            ["model", "dataset_id", "horizon", "MASE", "CRPS", "MASE_rank", "CRPS_rank"]
        metric: Metric to use for sorting. Defaults to "MASE"

    Returns:
        DataFrame: Leaderboard with:
            - MASE (norm.), CRPS (norm.): Geometric mean of Seasonal Naive-normalized values
            - MASE_rank, CRPS_rank: Average rank across configurations
            Sorted by the chosen metric.
    """
    if df_datasets.empty:
        return pd.DataFrame()

    if metric not in df_datasets.columns:
        return pd.DataFrame()

    # Step 1: Normalize MASE and CRPS by Seasonal Naive per (dataset_id, horizon)
    df_normalized = normalize_by_seasonal_naive(
        df_datasets,
        baseline_model=SEASONAL_NAIVE_MODEL,
        metrics=["MASE", "CRPS"],
        groupby_cols=["dataset_id", "horizon"],
    )

    if df_normalized.empty:
        print("❌ Error: Normalization failed. Make sure Seasonal Naive results are available.")
        return pd.DataFrame()

    # Step 2: Aggregate normalized MASE and CRPS with geometric mean
    def gmean_with_nan(x):
        """Compute geometric mean, ignoring NaN values."""
        valid = x.dropna()
        if len(valid) == 0:
            return np.nan
        return stats.gmean(valid)

    normalized_metrics = (
        df_normalized.groupby("model")[["MASE", "CRPS"]]
        .agg(gmean_with_nan)
        .reset_index()
    )

    # Rename columns
    normalized_metrics = normalized_metrics.rename(columns={
        "MASE": "MASE (norm.)",
        "CRPS": "CRPS (norm.)"
    })

    # Step 3: Compute average ranks from original data (pre-normalized)
    if "MASE_rank" in df_datasets.columns and "CRPS_rank" in df_datasets.columns:
        # Use the same configurations that were used in normalization
        df_with_baseline = df_datasets[
            df_datasets.set_index(["dataset_id", "horizon"]).index.isin(
                df_normalized.set_index(["dataset_id", "horizon"]).index.unique()
            )
        ]
        avg_ranks = (
            df_with_baseline.groupby("model")[["MASE_rank", "CRPS_rank"]]
            .mean()
            .reset_index()
        )
        # Merge normalized metrics with average ranks
        leaderboard = normalized_metrics.merge(avg_ranks, on="model", how="left")
    else:
        leaderboard = normalized_metrics

    # Step 4: Sort by chosen metric
    sort_metric = "MASE (norm.)" if metric == "MASE" else "CRPS (norm.)"

    if sort_metric in leaderboard.columns:
        leaderboard = leaderboard.sort_values(by=sort_metric, ascending=True).reset_index(drop=True)
    else:
        leaderboard = leaderboard.sort_values(by=leaderboard.columns[1], ascending=True).reset_index(drop=True)

    # Step 5: Select and order columns
    col_order = ["model", "MASE (norm.)", "CRPS (norm.)", "MASE_rank", "CRPS_rank"]
    col_order = [col for col in col_order if col in leaderboard.columns]
    leaderboard = leaderboard[col_order]
    leaderboard = leaderboard.round(3)

    return leaderboard


def extract_pollutant(item_id: str) -> str:
    """Extract pollutant name from item_id (e.g., 'site_105_..._IMD_CO' -> 'CO')."""
    return item_id.rsplit("_", 1)[-1]


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

    Sites where MASE > threshold for ANY model are excluded from ALL models
    to ensure a fair comparison.

    Returns:
        DataFrame with columns ["model", "dataset_id", "horizon", "pollutant", "MASE", "CRPS", "MAE", "RMSE"]
    """
    MASE_THRESHOLD = 50

    # --- Pass 1: collect per-site MASE across all models, then exclude by mean ---
    # Key: (dataset_id, horizon, item_id) -> list of MASE values across models
    site_mase_values: dict[tuple[str, str, str], list[float]] = {}

    for model_name, dataset_id, horizon, item_ids, npz_metrics in _iter_model_series(results_root, dataset_filter):
        n_series = len(item_ids)
        arr = npz_metrics.get("MASE")
        if arr is None or arr.shape[0] != n_series:
            continue
        reduce_axes = tuple(range(1, arr.ndim))
        per_series = np.nanmean(arr[:n_series], axis=reduce_axes) if reduce_axes else arr[:n_series]
        for i, iid in enumerate(item_ids):
            val = per_series[i]
            if not np.isnan(val):
                site_mase_values.setdefault((dataset_id, horizon, iid), []).append(float(val))

    # Exclude sites where mean MASE across models > threshold
    excluded_sites: dict[tuple[str, str], set[str]] = {}
    for (dataset_id, horizon, iid), values in site_mase_values.items():
        if np.mean(values) > MASE_THRESHOLD:
            excluded_sites.setdefault((dataset_id, horizon), set()).add(iid)

    # Log excluded sites with pollutant info
    if excluded_sites:
        print(f"\n  MASE threshold ({MASE_THRESHOLD}) exclusions (applied to ALL models):")
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
                per_series = np.nanmean(arr[:n_series], axis=reduce_axes) if reduce_axes else arr[:n_series]
                # Mask excluded sites across all metrics
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
    ].median()


def get_pollutant_balanced_leaderboard(
    pollutant_results: pd.DataFrame, metric: str = "MASE"
) -> pd.DataFrame:
    """
    Compute a pollutant-balanced overall leaderboard.

    1. Mean MASE/CRPS per (model, dataset, horizon, pollutant) — across sites
    2. Mean of those per (model, dataset, horizon) — balanced per-dataset score
    3. Normalize by Seasonal Naive's balanced score per (dataset, horizon)
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


def filter_by_datasets(df: pd.DataFrame, dataset_ids: list[str]) -> pd.DataFrame:
    """Filter results DataFrame to only include specified dataset_ids."""
    if not dataset_ids:
        raise ValueError("No dataset_ids provided")
    missing = set(dataset_ids) - set(df["dataset_id"].unique())
    if missing:
        raise ValueError(f"Datasets not found in results: {missing}")
    return df[df["dataset_id"].isin(dataset_ids)].copy()


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
    air_pollution = config_lb.get("air_pollution", False)

    print("=" * 80)
    print("TIME Local Leaderboard Calculator")
    print("=" * 80)
    if dataset_filter:
        print(f"  Filtering to datasets: {dataset_filter}")
    print()

    # Step 1: Load all results (including seasonal_naive) from results directory
    results_root = Path(results_dir)
    print(f"Step 1: Loading results from {results_root}...")

    if not results_root.exists():
        print(f"❌ Error: Results directory does not exist: {results_root}")
        sys.exit(1)

    all_results = get_all_datasets_results(results_root)

    if all_results.empty:
        print(f"❌ No results found in {results_root}")
        sys.exit(1)

    print(f"✅ Loaded {len(all_results)} results")

    # Filter to requested datasets
    if dataset_filter:
        all_results = filter_by_datasets(all_results, dataset_filter)
        print(f"   After filtering: {len(all_results)} results for {dataset_filter}")

    # Check that seasonal_naive is present
    if SEASONAL_NAIVE_MODEL not in all_results["model"].values:
        print(f"❌ No '{SEASONAL_NAIVE_MODEL}' results found in {results_root}.")
        print(f"   Run seasonal_naive first, or place its results in {results_root / SEASONAL_NAIVE_MODEL}/")
        sys.exit(1)

    # Step 2: Compute ranks
    print(f"\nStep 2: Computing ranks...")
    all_results = compute_ranks(all_results, groupby_cols=["dataset_id", "horizon"])

    print(f"   Models: {sorted(all_results['model'].unique())}")
    print(f"   Datasets: {sorted(all_results['dataset_id'].unique())}")

    # Export per-dataset raw results to CSV
    raw_csv = output_dir / "per_dataset_results.csv"
    all_results.round(4).to_csv(raw_csv, index=False)
    print(f"   Saved per-dataset results to {raw_csv}")

    if air_pollution:
        # --- Air pollution mode: per-pollutant + pollutant-balanced leaderboard ---
        print(f"\nStep 3: Computing per-pollutant leaderboard...")
        pollutant_results = get_per_pollutant_results(results_root, dataset_filter)

        if pollutant_results.empty:
            print("   No per-pollutant data available (item_ids missing from config.json?)")
            sys.exit(1)

        pollutants = sorted(pollutant_results["pollutant"].unique())
        print(f"   Found pollutants: {pollutants}")

        # Build per-pollutant tables: mean across sites per pollutant
        pollutant_agg_rows = []
        datasets_in_results = sorted(pollutant_results["dataset_id"].unique())
        for dataset_id in datasets_in_results:
            ddf = pollutant_results[pollutant_results["dataset_id"] == dataset_id]
            dataset_pollutants = sorted(ddf["pollutant"].unique())

            print(f"\n{'=' * 60}")
            print(f"  Dataset: {dataset_id}")
            print(f"{'=' * 60}")

            for pollutant in dataset_pollutants:
                pdf = ddf[ddf["pollutant"] == pollutant]
                agg = pdf.groupby("model")[["MASE", "CRPS", "MAE", "RMSE"]].mean().reset_index()
                agg = agg.sort_values(by=metric, ascending=True).reset_index(drop=True)
                agg = agg.round(4)

                print(f"\n  {'─' * 40}")
                print(f"    Pollutant: {pollutant}")
                print(f"  {'─' * 40}")
                print(agg.to_string(index=False))

                # Collect for CSV
                agg_csv = agg.copy()
                agg_csv.insert(0, "dataset_id", dataset_id)
                agg_csv.insert(1, "pollutant", pollutant)
                pollutant_agg_rows.append(agg_csv)

        print()

        # Pollutant-balanced overall leaderboard
        balanced_lb = get_pollutant_balanced_leaderboard(pollutant_results, metric=metric)
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

        # Export per-pollutant results to CSV
        if pollutant_agg_rows:
            pollutant_csv_df = pd.concat(pollutant_agg_rows, ignore_index=True)
            pollutant_csv = output_dir / "per_pollutant_leaderboard.csv"
            pollutant_csv_df.to_csv(pollutant_csv, index=False)
            print(f"   Saved per-pollutant leaderboard to {pollutant_csv}")

            raw_pollutant_csv = output_dir / "per_pollutant_results.csv"
            pollutant_results.round(4).to_csv(raw_pollutant_csv, index=False)
            print(f"   Saved raw per-pollutant results to {raw_pollutant_csv}")

    else:
        # --- Original mode: overall leaderboard ---
        print(f"\nStep 3: Computing Overall Leaderboard...")
        leaderboard = get_overall_leaderboard(all_results, metric=metric)

        if leaderboard.empty:
            print("Failed to compute leaderboard")
            sys.exit(1)

        print("\n" + "=" * 80)
        print("Overall Leaderboard")
        print("=" * 80)
        print()
        print(leaderboard.to_string(index=False))
        print()

        print("\n" + "=" * 80)
        print("Note: Metrics are normalized by Seasonal Naive baseline.")
        print("      Lower values are better. Seasonal Naive = 1.0")
        print("=" * 80)

        overall_csv = output_dir / "overall_leaderboard.csv"
        leaderboard.to_csv(overall_csv, index=False)
        print(f"\n   Saved overall leaderboard to {overall_csv}")


if __name__ == "__main__":
    main()