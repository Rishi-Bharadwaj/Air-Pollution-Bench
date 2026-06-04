#!/usr/bin/env python3
"""
Imputation Ablation Study — Leaderboard by imputation threshold.

For each threshold (5%, 10%, 15%, 20%, 30%), loads the allowed site lists
from imputation_ablation/<dataset>/lt_<X>pct.csv, then computes the same
leaderboard as compute_local_leaderboard but additionally masking out any
sites whose imputation % exceeds the threshold.

Output is written to imputation_ablation/results_lt_<X>pct/.
"""

import sys
from pathlib import Path
from time import sleep
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from compute_local_leaderboard import (
    _iter_model_series,
    get_pollutant_balanced_leaderboard,
)
from leaderboard_utils import extract_pollutant, display_dataset, to_latex_table, MODEL_GROUPS, GROUP_ORDER

# Map from imputation_ablation folder names to results folder names
DATASET_NAME_MAP = {
    "CNEMC": "CNEMC_SMALL",
}

THRESHOLDS = [5, 10, 15, 20, 30]

ABLATION_DIR = Path(__file__).parent / "imputation_ablation"
RESULTS_ROOT = Path(__file__).parent / "output" / "results"


def load_allowed_sites(threshold: int) -> dict[str, set[str]]:
    """
    Load allowed site names for a given threshold.

    Returns:
        dict mapping results dataset name (e.g. "CNEMC_SMALL") to set of item_ids (without .csv)
    """
    allowed = {}
    filename = f"lt_{threshold}pct.csv"
    for dataset_dir in ABLATION_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        csv_path = dataset_dir / filename
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        site_ids = set(df["file"].str.replace(".csv", "", regex=False))
        results_name = DATASET_NAME_MAP.get(dataset_dir.name, dataset_dir.name)
        allowed[results_name] = site_ids
    return allowed


def get_per_pollutant_results_ablation(
    results_root: Path,
    allowed_sites: dict[str, set[str]],
    dataset_filter: list[str] = None,
) -> pd.DataFrame:
    """
    Same as get_per_pollutant_results in compute_local_leaderboard, but adds
    an extra exclusion: sites not in allowed_sites for their dataset.
    """
    THRESHOLD = 50

    # --- Pass 1: collect per-site MASE and CRPS across all models ---
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

    # Exclude sites where mean MASE > threshold OR mean CRPS > threshold OR not in allowed list
    excluded_sites: dict[tuple[str, str], set[str]] = {}
    all_site_keys = set(site_metric_values["MASE"]) | set(site_metric_values["CRPS"])
    for key in all_site_keys:
        dataset_id, horizon, iid = key
        mase_vals = site_metric_values["MASE"].get(key, [])
        crps_vals = site_metric_values["CRPS"].get(key, [])
        dataset_name = dataset_id.split("/")[0]

        mase_excluded = mase_vals and np.mean(mase_vals) > THRESHOLD
        crps_excluded = crps_vals and np.mean(crps_vals) > THRESHOLD
        imputation_excluded = dataset_name in allowed_sites and iid not in allowed_sites[dataset_name]

        if mase_excluded or crps_excluded or imputation_excluded:
            excluded_sites.setdefault((dataset_id, horizon), set()).add(iid)

    if excluded_sites:
        print(f"\n  Exclusions (MASE/CRPS>{THRESHOLD} OR above imputation threshold):")
        for (ds, hz), ids in sorted(excluded_sites.items()):
            pollutant_counts: dict[str, int] = {}
            for iid in ids:
                pol = extract_pollutant(iid)
                pollutant_counts[pol] = pollutant_counts.get(pol, 0) + 1
            breakdown = ", ".join(f"{pol}: {n}" for pol, n in sorted(pollutant_counts.items()))
            print(f"    {ds}/{hz}: {len(ids)} site(s) excluded ({breakdown})")

    # --- Pass 2: load all metrics, masking excluded sites ---
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
                for i, iid in enumerate(item_ids):
                    if iid in exclude_ids:
                        per_series[i] = np.nan
                batch[metric_name] = per_series.tolist()
            else:
                batch[metric_name] = [np.nan] * n_series
                print(f"Error: missing {metric_name} for {model_name}/{dataset_id}/{horizon}")
                exit()
        rows.append(pd.DataFrame(batch))

    if not rows:
        return pd.DataFrame(columns=["model", "dataset_id", "horizon", "pollutant", "MASE", "CRPS", "MAE", "RMSE"])

    df = pd.concat(rows, ignore_index=True)
    return df.groupby(["model", "dataset_id", "horizon", "pollutant"], as_index=False)[
        ["MASE", "CRPS", "MAE", "RMSE"]
    ].mean()


def run_for_threshold(threshold: int, metric: str = "MASE"):
    """Run the full leaderboard computation for a single imputation threshold."""
    print(f"\n{'#' * 80}")
    print(f"# Imputation Ablation: threshold < {threshold}%")
    print(f"{'#' * 80}")

    allowed_sites = load_allowed_sites(threshold)
    if not allowed_sites:
        print(f"  No allowed-site files found for threshold {threshold}%")
        return

    for ds_name, sites in sorted(allowed_sites.items()):
        print(f"  {ds_name}: {len(sites)} sites allowed")

    output_dir = ABLATION_DIR / f"results_lt_{threshold}pct"
    output_dir.mkdir(parents=True, exist_ok=True)

    pollutant_results = get_per_pollutant_results_ablation(RESULTS_ROOT, allowed_sites)

    if pollutant_results.empty:
        print("   No per-pollutant data available")
        return

    pollutants = sorted(pollutant_results["pollutant"].unique())
    print(f"   Found pollutants: {pollutants}")

    # Per-pollutant per-dataset tables
    datasets_in_results = sorted(pollutant_results["dataset_id"].unique())
    pol_subdir = output_dir / "per_pollutant"
    pol_csv_subdir = output_dir / "per_pollutant_csv"
    pol_subdir.mkdir(parents=True, exist_ok=True)
    pol_csv_subdir.mkdir(parents=True, exist_ok=True)

    for dataset_id in datasets_in_results:
        ddf = pollutant_results[pollutant_results["dataset_id"] == dataset_id]
        dataset_pollutants = sorted(ddf["pollutant"].unique())

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
            agg = agg.round(4)

            print(f"\n  {'─' * 40}")
            print(f"    Pollutant: {pollutant}")
            print(f"  {'─' * 40}")
            print(agg.to_string(index=False))

            caption = f"{pollutant} leaderboard --- {display_dataset(dataset_id)}"
            tex = to_latex_table(agg, caption, metric_cols=["MASE", "CRPS", "MAE", "RMSE"],
                                 model_groups=MODEL_GROUPS, group_order=GROUP_ORDER)
            (dataset_subdir / f"{pollutant}.tex").write_text(tex)
            agg.to_csv(dataset_csv_subdir / f"{pollutant}.csv", index=False)

    # Overall balanced leaderboard
    balanced_lb = get_pollutant_balanced_leaderboard(
        pollutant_results, metric=metric,
        output_dir=output_dir,
        model_groups=MODEL_GROUPS,
        group_order=GROUP_ORDER,
    )
    if not balanced_lb.empty:
        print(f"\n{'=' * 60}")
        print(f"  Pollutant-Balanced Overall Leaderboard (< {threshold}% imputation)")
        print(f"{'=' * 60}")
        print(balanced_lb.to_string(index=False))
        print()

        balanced_csv = output_dir / "pollutant_balanced_leaderboard.csv"
        balanced_lb.to_csv(balanced_csv, index=False)
        print(f"   Saved to {balanced_csv}")

        balanced_tex = output_dir / "pollutant_balanced_leaderboard.tex"
        balanced_tex.write_text(to_latex_table(
            balanced_lb, f"Pollutant-balanced leaderboard (< {threshold}\\% imputation)",
            metric_cols=["MASE (norm.)", "CRPS (norm.)"],
            model_groups=MODEL_GROUPS, group_order=GROUP_ORDER,
        ))
        print(f"   Saved to {balanced_tex}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Imputation ablation leaderboard")
    parser.add_argument("--threshold", type=int, nargs="+", default=None,
                        help=f"Threshold(s) to run (default: all = {THRESHOLDS})")
    parser.add_argument("--metric", type=str, default="MASE", choices=["MASE", "CRPS"])
    args = parser.parse_args()

    thresholds = args.threshold or THRESHOLDS

    print("=" * 80)
    print("Imputation Ablation Study")
    print("=" * 80)

    for t in thresholds:
        run_for_threshold(t, metric=args.metric)
        sleep(10)

    print("\nDone.")


if __name__ == "__main__":
    main()