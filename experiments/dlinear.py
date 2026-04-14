"""
DLinear baseline experiments for time series forecasting.

This script uses AutoGluon-TimeSeries to train a DLinear model. A separate
model is trained per pollutant group (grouped by the trailing token of each
series' item_id, e.g. "site_ABD9_NO2" -> "NO2").

Training data is the full training split (everything before the test region).
At inference time, each test window is fed as an independent series so that
AutoGluon predicts from the end of each context window.

Usage:
    python experiments/dlinear.py
    python experiments/dlinear.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/dlinear.py --dataset all_datasets
"""

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms

# Load environment variables
load_dotenv()


def _entries_to_ag_df(entries, freq):
    """Convert GluonTS data entries to an AutoGluon long DataFrame."""
    anchor = pd.Timestamp("2000-01-01")
    frames = []
    for i, entry in enumerate(entries):
        y = np.asarray(entry["target"], dtype=float)
        ds = pd.date_range(anchor, periods=len(y), freq=freq)
        frames.append(pd.DataFrame({
            "item_id": str(i),
            "timestamp": ds,
            "target": y,
        }))
    return pd.concat(frames, ignore_index=True)


def run_dlinear_experiment(
    dataset_name: str,
    terms: list[str] = None,
    output_dir: str | None = None,
    context_length: int | None = None,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
):
    """
    Train one DLinear model per pollutant group and save quantile
    forecasts on the test split.
    """
    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = "./output/results/dlinear"
    os.makedirs(output_dir, exist_ok=True)

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'='*60}")
    print(f"Model: DLinear (AutoGluon)")
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
        num_windows = dataset.windows
        eval_data = dataset.test_data
        h = dataset.prediction_length

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - Test split: {test_length} steps")
        print(f"    - Prediction length: {h}")
        print(f"    - Windows: {num_windows}")
        print(f"    - Season length: {season_length}")

        effective_context_length = context_length if context_length is not None else h

        # Materialize training entries and test inputs
        training_entries = list(dataset.training_dataset)
        test_inputs = list(eval_data.input)
        num_series_exp = len(training_entries)

        expected_instances = num_series_exp * num_windows
        assert len(test_inputs) == expected_instances, (
            f"Expected {expected_instances} test instances "
            f"(num_series={num_series_exp} x num_windows={num_windows}), "
            f"got {len(test_inputs)}"
        )

        # Group series by pollutant (trailing token of item_id)
        pollutant_to_indices: dict[str, list[int]] = defaultdict(list)
        for s_idx, entry in enumerate(training_entries):
            item_id = str(entry.get("item_id", s_idx))
            pollutant = item_id.rsplit("_", 1)[-1] if "_" in item_id else item_id
            pollutant_to_indices[pollutant].append(s_idx)

        pollutant_summary = {p: len(ids) for p, ids in pollutant_to_indices.items()}
        print(f"  Pollutant groups ({len(pollutant_to_indices)}): {pollutant_summary}")

        num_q = len(quantile_levels)
        fc_quantiles = np.zeros((expected_instances, num_q, h), dtype=np.float32)

        pollutant_items = list(pollutant_to_indices.items())
        num_pollutants = len(pollutant_items)

        for p_idx, (pollutant, series_indices) in enumerate(pollutant_items, 1):
            print(
                f"\n  ===== [{p_idx}/{num_pollutants}] Training DLinear for "
                f"pollutant '{pollutant}' on {len(series_indices)} series ====="
            )

            # --- Build training DataFrame ---
            train_group = [training_entries[i] for i in series_indices]
            train_df = _entries_to_ag_df(train_group, dataset.freq)
            train_tsdf = TimeSeriesDataFrame.from_data_frame(
                train_df, id_column="item_id", timestamp_column="timestamp",
            )

            # --- Fit DLinear ---
            predictor = TimeSeriesPredictor(
                prediction_length=h,
                quantile_levels=quantile_levels,
                verbosity=1,
            )
            hyperparams = {
                "DLinear": {
                    "max_epochs": max_epochs,
                },
            }
            if early_stopping_patience > 0:
                hyperparams["DLinear"]["early_stopping_patience"] = early_stopping_patience
            if context_length is not None:
                hyperparams["DLinear"]["context_length"] = context_length

            t0 = time.perf_counter()
            predictor.fit(train_tsdf, hyperparameters=hyperparams)
            predictor.persist()
            train_elapsed = time.perf_counter() - t0
            print(f"  [{pollutant}] training done in {train_elapsed:.1f}s")

            del train_df, train_tsdf, train_group

            # --- Predict all test windows for this pollutant ---
            group_test_inputs = []
            dest_flat_indices = []
            for s_idx in series_indices:
                base = s_idx * num_windows
                for w in range(num_windows):
                    group_test_inputs.append(test_inputs[base + w])
                    dest_flat_indices.append(base + w)

            num_total = len(group_test_inputs)
            print(f"    [{pollutant}] Predicting {num_total} test windows...")

            q_cols = [str(q) for q in quantile_levels]

            all_inputs = [
                {**e, "target": np.asarray(e["target"])[-effective_context_length:]}
                for e in group_test_inputs
            ]
            pred_df = _entries_to_ag_df(all_inputs, dataset.freq)
            pred_tsdf = TimeSeriesDataFrame.from_data_frame(
                pred_df, id_column="item_id", timestamp_column="timestamp",
            )
            predictions = predictor.predict(pred_tsdf)

            # predictions is a TimeSeriesDataFrame indexed by (item_id, timestamp)
            # with columns "mean", "0.1", "0.2", ..., "0.9"
            pred_reset = predictions[q_cols].reset_index()
            pred_reset["_order"] = pred_reset["item_id"].astype(int)
            pred_reset = pred_reset.sort_values(["_order", "timestamp"])
            pred_vals = pred_reset[q_cols].to_numpy()  # (num_total * h, num_q)
            pred_vals = pred_vals.reshape(num_total, h, num_q)
            fc_quantiles[dest_flat_indices] = pred_vals.transpose(0, 2, 1)  # (num_total, num_q, h)

            del pred_df, pred_tsdf, predictions

            del predictor
            gc.collect()

        assert fc_quantiles.shape == (expected_instances, num_q, h)

        # Save results
        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model": "DLinear",
            "context_length": effective_context_length,
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
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
    parser = argparse.ArgumentParser(description="Run DLinear baseline experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Port_Activity/D"],
                        help="Dataset name(s) or 'all_datasets'")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Context length fed to DLinear. Defaults to prediction_length.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Epochs without val loss improvement before stopping (0 to disable)")
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
            run_dlinear_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                output_dir=args.output_dir,
                context_length=args.context_length,
                config_path=config_path,
                quantile_levels=args.quantiles,
                max_epochs=args.max_epochs,
                early_stopping_patience=args.early_stopping_patience,
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