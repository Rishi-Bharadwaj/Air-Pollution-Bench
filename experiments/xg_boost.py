"""
XGBoost experiments for time series forecasting (via mlforecast).

This script trains XGBoost quantile regressors per (dataset, term, pollutant)
combination using mlforecast. One independent XGBRegressor is trained per
quantile level using xgboost's native `reg:quantileerror` objective, yielding
a full quantile forecast without any post-hoc conformal step.

Because pollutants have very different scales, a separate set of XGBoost
models is trained per pollutant (grouped by the trailing token of each
series' item_id, e.g. "site_ABD9_NO2" -> "NO2"). Typical AQ datasets have
6 pollutants, so 6 MLForecast fits are done per (dataset, term).

The full training set is loaded into memory up front; no DataLoader workers
are used.

Usage:
    python experiments/xg_boost.py
    python experiments/xg_boost.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/xg_boost.py --dataset "SG_Weather/D" "SG_PM25/H"
    python experiments/xg_boost.py --dataset all_datasets
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from mlforecast import MLForecast
from xgboost import XGBRegressor

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms

# Load environment variables
load_dotenv()

SEED = 42
np.random.seed(SEED)


def _build_long_df(entries, freq: str) -> pd.DataFrame:
    """Convert GluonTS-style entries to an mlforecast long DataFrame.

    The absolute timestamps are synthetic -- mlforecast only needs a
    monotonic index at the correct frequency.
    """
    anchor = pd.Timestamp("2000-01-01")
    frames = []
    for i, entry in enumerate(entries):
        y = np.asarray(entry["target"], dtype=float)
        ds = pd.date_range(anchor, periods=len(y), freq=freq)
        frames.append(pd.DataFrame({
            "unique_id": str(i),
            "ds": ds,
            "y": y,
        }))
    return pd.concat(frames, ignore_index=True)


def _default_lags(season_length: int) -> list[int]:
    """Reasonable lag set given the detected seasonality."""
    base = [1, 2, 3, 4, 5, 6, 7]
    if season_length and season_length > 1:
        # Add multiples of the seasonal period to capture seasonality.
        seasonal = [season_length, 2 * season_length]
        return sorted(set(base + seasonal))
    return base


def run_xgboost_experiment(
    dataset_name: str,
    terms: list[str] = None,
    output_dir: str | None = None,
    context_length: int | None = None,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    n_estimators: int = 500,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: float = 1.0,
    num_threads: int = 1,
    device: str = "cpu",
    lags: list[int] | None = None,
):
    """
    Train one set of XGBoost quantile regressors per pollutant and save
    quantile forecasts on the test split.

    Args:
        dataset_name: Dataset name (e.g., "SG_Weather/D")
        terms: List of terms to evaluate ("short", "medium", "long")
        output_dir: Output directory for results
        context_length: If set, crop each training/test series to the
            trailing `context_length` points before training/forecasting.
        config_path: Path to datasets.yaml config file
        quantile_levels: List of quantile levels to forecast
        n_estimators: Number of boosting rounds per quantile model
        max_depth: XGBoost tree depth
        learning_rate: XGBoost learning rate
        subsample: Row subsample ratio
        colsample_bytree: Column subsample ratio
        min_child_weight: Minimum sum of instance weight in a child
        num_threads: mlforecast feature-engineering threads
        device: XGBoost device ("cpu" or "cuda")
        lags: Explicit lag list. If None, picked from seasonality.
    """
    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = "./output/results/xg_boost"
    os.makedirs(output_dir, exist_ok=True)

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'='*60}")
    print(f"Model: XGBoost (mlforecast, quantile regression)")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # XGBoost on mlforecast is univariate -- convert multivariate datasets.
        to_univariate = False if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim == 1 else True

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        season_length = get_seasonality(dataset.freq)
        effective_lags = lags if lags is not None else _default_lags(season_length)

        num_windows = dataset.windows
        eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - Test split: {test_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")
        print(f"    - Season length: {season_length}")
        print(f"    - Lags: {effective_lags}")

        # Materialize per-series training entries and per-series test inputs.
        training_entries = list(dataset.training_dataset)
        test_inputs = list(eval_data.input)
        num_series_exp = len(training_entries)
        expected_instances = num_series_exp * num_windows
        assert len(test_inputs) == expected_instances, (
            f"Expected {expected_instances} test instances "
            f"(num_series={num_series_exp} * num_windows={num_windows}), "
            f"got {len(test_inputs)}"
        )

        # Optional context-length crop (applied to both training and test).
        def _crop(entries):
            if context_length is None:
                return entries
            cropped = []
            for e in entries:
                ce = dict(e)
                ce["target"] = np.asarray(e["target"])[-context_length:]
                cropped.append(ce)
            return cropped

        training_entries = _crop(training_entries)
        test_inputs = _crop(test_inputs)

        # Group series by pollutant (trailing token of item_id).
        pollutant_to_indices: dict[str, list[int]] = defaultdict(list)
        for s_idx, entry in enumerate(training_entries):
            item_id = str(entry.get("item_id", s_idx))
            pollutant = item_id.rsplit("_", 1)[-1] if "_" in item_id else item_id
            pollutant_to_indices[pollutant].append(s_idx)

        pollutant_summary = {p: len(ids) for p, ids in pollutant_to_indices.items()}
        print(f"  Pollutant groups ({len(pollutant_to_indices)}): {pollutant_summary}")

        h = dataset.prediction_length
        num_q = len(quantile_levels)
        fc_quantiles = np.zeros((expected_instances, num_q, h), dtype=np.float32)

        quantile_cols = [f"XGBoost_q{int(round(q * 100))}" for q in quantile_levels]

        pollutant_items = list(pollutant_to_indices.items())
        num_pollutants = len(pollutant_items)
        for p_idx, (pollutant, series_indices) in enumerate(pollutant_items, 1):
            print(
                f"\n  ===== [{p_idx}/{num_pollutants}] Training XGBoost for "
                f"pollutant '{pollutant}' on {len(series_indices)} series ====="
            )

            train_group = [training_entries[i] for i in series_indices]
            train_df = _build_long_df(train_group, dataset.freq)

            # One XGBRegressor per quantile level.
            models = {
                f"XGBoost_q{int(round(q * 100))}": XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    tree_method="hist",
                    device=device,
                    verbosity=0,
                    random_state=SEED,
                )
                for q in quantile_levels
            }

            mlf = MLForecast(
                models=models,
                freq=dataset.freq,
                lags=effective_lags,
                num_threads=num_threads,
            )

            t0 = time.perf_counter()
            mlf.fit(df=train_df)
            train_elapsed = time.perf_counter() - t0
            print(f"  [{pollutant}] training done in {train_elapsed:.1f}s")

            # Gather this pollutant's test windows in the same series order.
            group_test_inputs = []
            dest_flat_indices = []
            for s_idx in series_indices:
                base = s_idx * num_windows
                for w in range(num_windows):
                    group_test_inputs.append(test_inputs[base + w])
                    dest_flat_indices.append(base + w)

            # Build a "new_df" with one unique_id per test window. mlforecast
            # will use each series' trailing history to build lag features and
            # recursively forecast `h` steps.
            new_df = _build_long_df(group_test_inputs, dataset.freq)
            # Rename unique_ids to avoid collision with training ids (strings).
            id_map = {str(i): f"w{i}" for i in range(len(group_test_inputs))}
            new_df["unique_id"] = new_df["unique_id"].map(id_map)

            print(
                f"    [{pollutant}] Predicting {len(group_test_inputs)} test windows..."
            )
            t0 = time.perf_counter()
            pred_df = mlf.predict(h=h, new_df=new_df)
            predict_elapsed = time.perf_counter() - t0
            print(f"    [{pollutant}] prediction done in {predict_elapsed:.1f}s")

            # Place each window's forecast into the global buffer by looking
            # up its unique_id. We don't rely on pred_df row order since the
            # ids are strings and sort lexicographically ("w10" < "w2").
            inv_id_map = {v: int(k) for k, v in id_map.items()}
            for uid, grp in pred_df.groupby("unique_id", sort=False):
                grp = grp.sort_values("ds")
                # (h, num_quantiles) -> (num_quantiles, h)
                q_arr = grp[quantile_cols].to_numpy().astype(np.float32).T
                original_pos = inv_id_map[uid]
                dest_idx = dest_flat_indices[original_pos]
                fc_quantiles[dest_idx] = q_arr

        assert fc_quantiles.shape == (expected_instances, num_q, h), (
            f"Unexpected forecast shape {fc_quantiles.shape}, "
            f"expected ({expected_instances}, {num_q}, {h})"
        )

        ds_config = f"{dataset_name}/{term}"

        model_hyperparams = {
            "model": "XGBoost",
            "library": "mlforecast",
            "context_length": context_length,
            "prediction_length": dataset.prediction_length,
            "lags": effective_lags,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "device": device,
            "season_length": season_length,
            "quantile_levels": quantile_levels,
        }

        metadata = save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
            ds_config=ds_config,
            output_base_dir=output_dir,
            seasonality=season_length,
            model_hyperparams=model_hyperparams,
            quantile_levels=quantile_levels,
        )
        print(f"  Completed: {metadata['num_series']} series x {metadata['num_windows']} windows")
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run XGBoost (mlforecast) experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Port_Activity/D"],
                        help="Dataset name(s). Single dataset, multiple datasets, or 'all_datasets'.")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Optional crop: keep only last N timesteps of each series.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--n-estimators", type=int, default=500, help="Boosting rounds per quantile")
    parser.add_argument("--max-depth", type=int, default=8, help="XGBoost tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="XGBoost learning rate")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsample ratio")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsample ratio")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="Min child weight")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="mlforecast feature-engineering threads")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="XGBoost device")
    parser.add_argument("--lags", type=int, nargs="+", default=None,
                        help="Explicit lag list (defaults to seasonality-based set)")
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Quantile levels to save")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    total_datasets = len(datasets)
    failed = False
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_xgboost_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                output_dir=args.output_dir,
                context_length=args.context_length,
                config_path=config_path,
                quantile_levels=args.quantiles,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                min_child_weight=args.min_child_weight,
                num_threads=args.num_threads,
                device=args.device,
                lags=args.lags,
            )
        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            failed = True
            continue

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
