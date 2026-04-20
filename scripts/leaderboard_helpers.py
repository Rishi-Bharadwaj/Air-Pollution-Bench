import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path to import timebench utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
SEASONAL_NAIVE_MODEL = "seasonal_naive"

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


