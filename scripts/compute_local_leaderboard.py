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


_DATASET_NAME_MAP = {
    "CNEMC": "CNEMC SMALL",
}


def display_dataset(dataset_id: str) -> str:
    """Return a human-readable dataset name, stripping frequency suffix and mapping aliases.

    E.g. 'CPCB/H' -> 'CPCB', 'CNEMC/H' -> 'CNEMC SMALL', 'MY_DS/D' -> 'MY DS'
    """
    name = dataset_id.split("/")[0]
    name = _DATASET_NAME_MAP.get(name, name)
    return name.replace("_", " ")


def to_latex_table(
    df: pd.DataFrame,
    caption: str,
    table_num: int,
    metric_cols: list = None,
    lower_is_better: bool = True,
) -> str:
    """
    Convert a DataFrame to a LaTeX table snippet (suitable for \\input{}).

    Formatting per metric column:
      - Bold:      best value
      - Underline: second best
      - Italics:   third best

    Caption is placed above the table in 9pt type, centered if it fits on
    one line (<= 60 chars), otherwise flush left, with 0.1in spacing before
    and after. Requires booktabs in the parent document.
    """
    df = df.reset_index(drop=True)
    if metric_cols is None:
        metric_cols = [c for c in df.columns if c != "model"]

    # Determine rank-based formatting per metric column
    cell_fmt: dict[tuple[int, str], str] = {}
    for col in metric_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        sorted_idx = vals.sort_values(ascending=lower_is_better).dropna().index.tolist()
        for rank, idx in enumerate(sorted_idx[:3]):
            cell_fmt[(idx, col)] = ["bold", "underline", "italic"][rank]

    def _escape(s: str) -> str:
        # Escape all LaTeX special characters in order (backslash first)
        s = s.replace("\\", "\\textbackslash{}")
        s = s.replace("{", "\\{").replace("}", "\\}")
        s = s.replace("$", "\\$").replace("#", "\\#")
        s = s.replace("^", "\\textasciicircum{}")
        s = s.replace("~", "\\textasciitilde{}")
        s = s.replace("_", "\\_")
        s = s.replace("%", "\\%").replace("&", "\\&")
        s = s.replace("<", "\\textless{}").replace(">", "\\textgreater{}")
        return s

    def _fmt(val, fmt, is_str=False):
        s = _escape(str(val)) if is_str else str(val)
        if fmt == "bold":
            return f"\\textbf{{{s}}}"
        if fmt == "underline":
            return f"\\underline{{{s}}}"
        if fmt == "italic":
            return f"\\textit{{{s}}}"
        return s

    cols = df.columns.tolist()
    col_spec = "l" + "r" * (len(cols) - 1)
    header = " & ".join(f"\\textbf{{{_escape(c)}}}" for c in cols) + " \\\\"

    body_lines = []
    for idx, row in df.iterrows():
        cells = [_fmt(row[c], cell_fmt.get((idx, c)), is_str=(c not in metric_cols)) for c in cols]
        body_lines.append(" & ".join(cells) + " \\\\")

    caption_align = "centering" if len(caption) <= 60 else "raggedright"
    caption_tex = (
        f"{{\\fontsize{{9}}{{11}}\\selectfont\\{caption_align}"
        f" \\textit{{Table~{table_num}:}} {_escape(caption)}\\par}}"
    )

    return "\n".join([
        "\\begin{table}[h]",
        "\\vspace{0.1in}",
        caption_tex,
        "\\vspace{0.1in}",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
        *body_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


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


def check_result_consistency(results_root: Path, dataset_filter: list[str] = None) -> bool:
    """
    Check that all models and datasets have the same number of results per item_id/series.

    For each (dataset_id, horizon), verifies:
      - All models have the same set of item_ids (in the same order).
      - All models have the same NPZ array shape per metric (same windows/variates per series).

    Prints warnings for any inconsistencies found.

    Returns:
        True if all checks pass, False if any inconsistencies are found.
    """
    import json

    # Collect: (dataset_id, horizon) -> {model: {"item_ids": [...], "shapes": {metric: shape}}}
    registry: dict[tuple[str, str], dict[str, dict]] = {}

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
                    if not metrics_path.exists():
                        continue
                    item_ids = None
                    if config_path.exists():
                        with open(config_path) as f:
                            cfg = json.load(f)
                        item_ids = cfg.get("item_ids")
                    npz = np.load(metrics_path)
                    shapes = {k: npz[k].shape for k in npz.files}
                    key = (dataset_id, horizon)
                    registry.setdefault(key, {})[model_name] = {
                        "item_ids": item_ids,
                        "shapes": shapes,
                    }

    if not registry:
        print("⚠️  Consistency check: no results found to check.")
        return True

    all_ok = True
    for (dataset_id, horizon), model_data in sorted(registry.items()):
        models = sorted(model_data.keys())
        if len(models) < 2:
            continue

        # --- Check item_ids consistency ---
        ref_model = models[0]
        ref_item_ids = model_data[ref_model]["item_ids"]
        for m in models[1:]:
            m_item_ids = model_data[m]["item_ids"]
            if ref_item_ids is None and m_item_ids is None:
                continue
            if ref_item_ids != m_item_ids:
                all_ok = False
                if ref_item_ids is None or m_item_ids is None:
                    print(
                        f"⚠️  Consistency [{dataset_id}/{horizon}]: "
                        f"'{ref_model}' has item_ids={ref_item_ids is not None}, "
                        f"'{m}' has item_ids={m_item_ids is not None}"
                    )
                else:
                    ref_set = set(ref_item_ids)
                    m_set = set(m_item_ids)
                    only_ref = ref_set - m_set
                    only_m = m_set - ref_set
                    print(
                        f"⚠️  Consistency [{dataset_id}/{horizon}]: item_ids differ between "
                        f"'{ref_model}' and '{m}'"
                        + (f" — only in '{ref_model}': {sorted(only_ref)}" if only_ref else "")
                        + (f" — only in '{m}': {sorted(only_m)}" if only_m else "")
                        + (f" — same items but different order" if not only_ref and not only_m else "")
                    )

        # --- Check array shapes consistency ---
        ref_shapes = model_data[ref_model]["shapes"]
        for m in models[1:]:
            m_shapes = model_data[m]["shapes"]
            for metric in set(ref_shapes) | set(m_shapes):
                ref_shape = ref_shapes.get(metric)
                m_shape = m_shapes.get(metric)
                if ref_shape != m_shape:
                    all_ok = False
                    print(
                        f"⚠️  Consistency [{dataset_id}/{horizon}] metric={metric}: "
                        f"'{ref_model}' shape={ref_shape}, '{m}' shape={m_shape}"
                    )

    if all_ok:
        print("✅ Consistency check passed: all models have matching item_ids and result shapes.")
    else:
        print("❌ Consistency check FAILED: see warnings above.")
    return all_ok


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

    # Consistency check: all models must have the same item_ids and result shapes
    print(f"\nStep 1b: Checking result consistency across models and datasets...")
    check_result_consistency(results_root, dataset_filter)

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
        pol_subdir = output_dir / "per_pollutant"
        pol_subdir.mkdir(parents=True, exist_ok=True)
        table_num = 1
        datasets_in_results = sorted(pollutant_results["dataset_id"].unique())
        for dataset_id in datasets_in_results:
            ddf = pollutant_results[pollutant_results["dataset_id"] == dataset_id]
            dataset_pollutants = sorted(ddf["pollutant"].unique())

            # Subfolder per dataset, using only the dataset name (no freq suffix)
            dataset_subdir = pol_subdir / dataset_id.split("/")[0]
            dataset_subdir.mkdir(parents=True, exist_ok=True)

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


                # Save individual LaTeX table
                caption = f"{pollutant} leaderboard --- {display_dataset(dataset_id)}"
                tex = to_latex_table(agg, caption, table_num, metric_cols=["MASE", "CRPS", "MAE", "RMSE"])
                pol_tex = dataset_subdir / f"{pollutant}.tex"
                pol_tex.write_text(tex)
                table_num += 1

                # Collect for combined CSV
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

            balanced_tex = output_dir / "pollutant_balanced_leaderboard.tex"
            balanced_caption = "Pollutant-balanced overall leaderboard (normalized by Seasonal Naive, gmean across datasets)"
            balanced_tex.write_text(to_latex_table(
                balanced_lb, balanced_caption, 2,
                metric_cols=["MASE (norm.)", "CRPS (norm.)"],
            ))
            print(f"   Saved pollutant-balanced LaTeX table to {balanced_tex}")

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

        overall_tex = output_dir / "overall_leaderboard.tex"
        overall_tex.write_text(to_latex_table(
            leaderboard, "Overall leaderboard (normalized by Seasonal Naive, gmean across datasets)", 1,
            metric_cols=["MASE (norm.)", "CRPS (norm.)"],
        ))
        print(f"\n   Saved overall leaderboard LaTeX table to {overall_tex}")


if __name__ == "__main__":
    main()