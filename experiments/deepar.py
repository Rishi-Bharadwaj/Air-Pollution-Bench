"""
DeepAR experiments for time series forecasting.

This script trains a DeepAR model (GluonTS PyTorch implementation) per
(dataset, term, series) combination using the training split, then produces
probabilistic forecasts on the test split. Quantile forecasts are obtained
directly from DeepAR's sample paths.

Because pollutants have very different scales, a separate DeepAR model is
trained per pollutant (grouped by the trailing token of each series'
item_id, e.g. "site_ABD9_NO2" → "NO2"). Typical AQ datasets have 6
pollutants (CO, NO2, Ozone, PM10, PM2.5, SO2), so 6 models are trained per
(dataset, term). No validation set is used — training runs for a fixed
number of epochs per pollutant group.

Usage:
    python experiments/deepar.py
    python experiments/deepar.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/deepar.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/deepar.py --dataset all_datasets  # Run all datasets from config
"""

import argparse
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

# Silence a noisy GluonTS 0.15 + PyTorch 2.9 deprecation warning
# (gluonts/torch/util.py uses list-based multidim indexing).
warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing",
    category=UserWarning,
)

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from gluonts.torch.model.deepar import DeepAREstimator
from lightning.pytorch import seed_everything

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
seed_everything(SEED, workers=True)


def run_deepar_experiment(
    dataset_name: str,
    terms: list[str] = None,
    output_dir: str | None = None,
    context_length: int | None = None,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    num_layers: int = 2,
    hidden_size: int = 40,
    dropout_rate: float = 0.1,
    max_epochs: int = 50,
    num_batches_per_epoch: int = 50,
    batch_size: int = 32,
    num_samples: int = 100,
    lr: float = 1e-3,
):
    """
    Train one DeepAR model per series (per pollutant) and save quantile
    forecasts on the test split. No validation set is used.

    Args:
        dataset_name: Dataset name (e.g., "SG_Weather/D")
        terms: List of terms to evaluate ("short", "medium", "long")
        output_dir: Output directory for results
        context_length: Context length fed to DeepAR. Defaults to prediction_length
            if None (GluonTS default).
        config_path: Path to datasets.yaml config file
        num_layers: Number of RNN layers
        hidden_size: Hidden size of the RNN
        dropout_rate: Dropout rate
        max_epochs: Max training epochs
        num_batches_per_epoch: Number of training batches per epoch
        batch_size: Training/inference batch size
        num_samples: Number of sample paths drawn at prediction time
        lr: Learning rate
    """
    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = "./output/results/deepar"

    os.makedirs(output_dir, exist_ok=True)

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'='*60}")
    print(f"Model: DeepAR")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # DeepAR is univariate — convert multivariate datasets
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

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - Test split: {test_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")
        print(f"    - Season length: {season_length}")

        effective_context_length = (
            context_length if context_length is not None else dataset.prediction_length
        )

        # Materialize per-series training entries and per-series test inputs.
        # eval_data.input is ordered series-major: all num_windows windows for
        # series 0, then all windows for series 1, etc. (matching what
        # save_window_predictions expects via idx = series*num_windows + window).
        training_entries = list(dataset.training_dataset)
        test_inputs = list(eval_data.input)
        num_series_exp = len(training_entries)

        expected_instances = num_series_exp * num_windows
        assert len(test_inputs) == expected_instances, (
            f"Expected {expected_instances} test instances "
            f"(num_series={num_series_exp} * num_windows={num_windows}), "
            f"got {len(test_inputs)}"
        )

        # Group series indices by pollutant. Pollutant is the trailing token of
        # item_id (e.g. "site_ABD9_NO2" -> "NO2"). Datasets that don't follow
        # this convention (e.g. SG_PM25 with item_id "raw") degenerate to a
        # single group containing all series, which is the correct behaviour.
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

        pollutant_items = list(pollutant_to_indices.items())
        num_pollutants = len(pollutant_items)
        for p_idx, (pollutant, series_indices) in enumerate(pollutant_items, 1):
            print(
                f"\n  ===== [{p_idx}/{num_pollutants}] Training DeepAR for "
                f"pollutant '{pollutant}' on {len(series_indices)} series "
                f"({max_epochs} epochs × {num_batches_per_epoch} batches) ====="
            )

            # Training data: all series belonging to this pollutant.
            train_group = [training_entries[i] for i in series_indices]

            # Fresh estimator per pollutant to reset weights and optimizer state.
            estimator = DeepAREstimator(
                freq=dataset.freq,
                prediction_length=dataset.prediction_length,
                context_length=effective_context_length,
                num_layers=num_layers,
                hidden_size=hidden_size,
                dropout_rate=dropout_rate,
                lr=lr,
                num_parallel_samples=num_samples,
                batch_size=batch_size,
                num_batches_per_epoch=num_batches_per_epoch,
                trainer_kwargs={
                    "max_epochs": max_epochs,
                    "enable_progress_bar": True,
                    "enable_model_summary": False,
                    "logger": False,
                    # NOTE: do not set enable_checkpointing=False — GluonTS
                    # always injects its own ModelCheckpoint callback, and
                    # Lightning raises if the two disagree.
                },
            )
            t0 = time.perf_counter()
            predictor = estimator.train(training_data=train_group)
            train_elapsed = time.perf_counter() - t0
            print(f"  [{pollutant}] training done in {train_elapsed:.1f}s")

            # Gather this pollutant's test windows in the same series order so
            # the forecast iterator lines up with our destination indices.
            group_test_inputs = []
            dest_flat_indices = []
            for s_idx in series_indices:
                base = s_idx * num_windows
                for w in range(num_windows):
                    group_test_inputs.append(test_inputs[base + w])
                    dest_flat_indices.append(base + w)

            print(
                f"    [{pollutant}] Predicting {len(group_test_inputs)} test windows..."
            )
            forecast_it = predictor.predict(group_test_inputs, num_samples=num_samples)

            for dest_idx, forecast in zip(dest_flat_indices, forecast_it):
                # forecast.quantile(q) returns an (h,) array
                q_arr = np.stack(
                    [np.asarray(forecast.quantile(q), dtype=np.float32) for q in quantile_levels],
                    axis=0,
                )  # (num_quantiles, h)
                fc_quantiles[dest_idx] = q_arr

        assert fc_quantiles.shape == (expected_instances, num_q, h), (
            f"Unexpected forecast shape {fc_quantiles.shape}, "
            f"expected ({expected_instances}, {num_q}, {h})"
        )

        # Save results
        ds_config = f"{dataset_name}/{term}"

        model_hyperparams = {
            "model": "DeepAR",
            "context_length": effective_context_length,
            "prediction_length": dataset.prediction_length,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "dropout_rate": dropout_rate,
            "lr": lr,
            "max_epochs": max_epochs,
            "num_batches_per_epoch": num_batches_per_epoch,
            "batch_size": batch_size,
            "num_samples": num_samples,
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
    parser = argparse.ArgumentParser(description="Run DeepAR experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Port_Activity/D"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets' to run all datasets from config")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Context length fed to DeepAR. Defaults to prediction_length.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--hidden-size", type=int, default=40, help="RNN hidden size")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-epochs", type=int, default=50, help="Max training epochs per pollutant")
    parser.add_argument("--num-batches-per-epoch", type=int, default=50,
                        help="Number of training batches per epoch")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of sample paths drawn at prediction time")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Quantile levels to save")
    args = parser.parse_args()

    # Handle dataset list or 'all_datasets'
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
            run_deepar_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                output_dir=args.output_dir,
                context_length=args.context_length,
                config_path=config_path,
                quantile_levels=args.quantiles,
                num_layers=args.num_layers,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout_rate,
                max_epochs=args.max_epochs,
                num_batches_per_epoch=args.num_batches_per_epoch,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                lr=args.lr,
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