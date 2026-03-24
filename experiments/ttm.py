"""
TTM (Tiny Time Mixer) model experiments for time series forecasting.
Uses conformal prediction to generate quantile forecasts from TTM's point predictions.

Usage:
    python experiments/ttm.py
    python experiments/ttm.py --context-length 512 --forecast-length 96
    python experiments/ttm.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/ttm.py --dataset all_datasets
"""

import argparse
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from transformers import Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.toolkit.get_model import get_model

# Map TIME benchmark frequency strings to pandas offset aliases
_FREQ_TO_PANDAS = {
    "H": "h", "D": "D", "W": "W", "M": "MS", "Q": "QS", "Y": "YS",
    "T": "min", "S": "s", "15T": "15min", "30T": "30min", "B": "B",
}

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms, clean_nan_target

load_dotenv()

SEED = 42
set_seed(SEED)


def conformal_quantiles_from_residuals(
    cal_preds: np.ndarray,
    cal_true: np.ndarray,
    test_preds: np.ndarray,
    quantile_levels: list[float],
) -> np.ndarray:
    """
    Generate quantile forecasts from point predictions using conformal prediction.

    Uses validation/calibration set residuals to empirically estimate the error
    distribution, then applies those error quantiles to test predictions.

    Args:
        cal_preds:  (n_cal, prediction_length) — point forecasts on calibration set
        cal_true:   (n_cal, prediction_length) — ground truth on calibration set
        test_preds: (n_test, prediction_length) — point forecasts on test set
        quantile_levels: list of quantile levels, e.g. [0.1, 0.2, ..., 0.9]

    Returns:
        fc_quantiles: (n_test, num_quantiles, prediction_length)
    """
    # Signed residuals: positive means under-prediction
    residuals = cal_true - cal_preds  # (n_cal, prediction_length)

    # Compute quantile corrections per horizon step
    corrections = np.quantile(
        residuals, quantile_levels, axis=0
    )  # (num_quantiles, prediction_length)

    # Apply corrections to test predictions
    # test_preds:  (n_test, pred_len) -> (n_test, 1, pred_len)
    # corrections: (num_quantiles, pred_len) -> (1, num_quantiles, pred_len)
    fc_quantiles = test_preds[:, None, :] + corrections[None, :, :]
    # (n_test, num_quantiles, prediction_length)

    return fc_quantiles.astype(np.float32)


def _extract_series_from_dataset(dataset, split="test"):
    """
    Extract time series arrays from a TIME Dataset object for TTM processing.

    Returns:
        contexts: list of np.ndarray, each (series_length,) for univariate
        targets:  list of np.ndarray, each (prediction_length,) ground truth
    """
    if split == "test":
        eval_data = dataset.test_data
    elif split == "val":
        eval_data = dataset.val_data
    else:
        raise ValueError(f"Unknown split: {split}")

    contexts = []
    targets = []

    for inp, label in eval_data:
        context = clean_nan_target(np.asarray(inp["target"]))
        if context.ndim > 1:
            context = context.squeeze(0)
        contexts.append(context)

        target = np.asarray(label["target"])
        if target.ndim > 1:
            target = target.squeeze(0)
        targets.append(target)

    return contexts, targets


def _run_ttm_inference(
    model_path: str,
    contexts: list[np.ndarray],
    context_length: int,
    prediction_length: int,
    batch_size: int,
    freq: str,
) -> np.ndarray:
    """
    Run TTM zero-shot inference using the Trainer API (handles freq_token automatically).

    Each context array is turned into a tiny single-series DataFrame, then
    TimeSeriesPreprocessor + Trainer.predict() produce the point forecasts.

    Returns:
        predictions: (n_instances, prediction_length) point forecasts
    """
    # Load TTM model
    model = get_model(
        model_path,
        context_length=context_length,
        prediction_length=prediction_length,
        freq_prefix_tuning=False,
        freq=None,
        prefer_l1_loss=False,
        prefer_longer_context=True,
    )

    pd_freq = _FREQ_TO_PANDAS.get(freq, freq)

    # Build a DataFrame with one "series" per context window.
    # Each series has context_length + prediction_length rows so that
    # TimeSeriesPreprocessor creates exactly 1 window per series.
    # Future values are dummy zeros (only the context matters for zero-shot).
    records = []
    for i, ctx in enumerate(contexts):
        ctx_arr = np.asarray(ctx, dtype=np.float64)
        # Crop or pad to context_length
        if len(ctx_arr) > context_length:
            ctx_arr = ctx_arr[-context_length:]
        elif len(ctx_arr) < context_length:
            ctx_arr = np.pad(ctx_arr, (context_length - len(ctx_arr), 0), mode="constant")

        # Append dummy future values for window creation
        full = np.concatenate([ctx_arr, np.zeros(prediction_length)])
        ts = pd.date_range("2020-01-01", periods=len(full), freq=pd_freq)

        for j in range(len(full)):
            records.append({"id": str(i), "date": ts[j], "value": full[j]})

    df = pd.DataFrame(records)

    tsp = TimeSeriesPreprocessor(
        timestamp_column="date",
        id_columns=["id"],
        target_columns=["value"],
        control_columns=[],
        context_length=context_length,
        prediction_length=prediction_length,
        scaling=False,
        encode_categorical=False,
    )

    split_config = {"train": 1.0, "test": 0.0}
    dset_train, _, _ = get_datasets(
        tsp, df, split_config,
        use_frequency_token=model.config.resolution_prefix_tuning,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            per_device_eval_batch_size=batch_size,
            seed=SEED,
            report_to="none",
        ),
    )

    print(f"    Running Trainer.predict() on {len(contexts)} instances...")
    preds = trainer.predict(dset_train).predictions[0]
    # Shape: (n_instances, prediction_length, n_channels)
    if preds.ndim == 3:
        preds = preds[:, :, 0]  # univariate → (n_instances, prediction_length)

    return preds


def run_ttm_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
    output_dir: str | None = None,
    batch_size: int = 64,
    context_length: int = 512,
    quantile_levels: list[float] | None = None,
    config_path: Path | None = None,
):
    """
    Run TTM zero-shot experiments on a dataset with conformal prediction quantiles.
    """
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = "./output/results/ttm_r2"

    os.makedirs(output_dir, exist_ok=True)

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"Model: {model_path}")
    print(f"Context length: {context_length}")
    print(f"Quantile method: Conformal Prediction")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        # Clamp prediction_length to TTM's max supported
        # Granite-TTM-R2 supports up to 720
        max_ttm_prediction = 720
        effective_pred_len = min(prediction_length, max_ttm_prediction)
        if effective_pred_len != prediction_length:
            print(
                f"  WARNING: prediction_length {prediction_length} exceeds TTM max "
                f"{max_ttm_prediction}, clamping to {effective_pred_len}"
            )

        print(
            f"  Config: prediction_length={prediction_length}, "
            f"test_length={test_length}, val_length={val_length}"
        )

        # TTM is univariate — always convert to univariate
        to_univariate = (
            False
            if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim == 1
            else True
        )

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Windows: {dataset.windows}")
        print(f"    - Prediction length: {dataset.prediction_length}")

        freq_str = str(dataset.freq)
        pd_freq = _FREQ_TO_PANDAS.get(freq_str, freq_str)
        season_length = get_seasonality(pd_freq)

        # --- Step 1: Run inference on validation set (for conformal calibration) ---
        print("    Running inference on validation set (for conformal calibration)...")
        val_contexts, val_targets = _extract_series_from_dataset(dataset, split="val")

        if len(val_contexts) > 0:
            val_preds = _run_ttm_inference(
                model_path=model_path,
                contexts=val_contexts,
                context_length=context_length,
                prediction_length=effective_pred_len,
                batch_size=batch_size,
                freq=freq_str,
            )
            val_true = np.stack(val_targets)[:, :effective_pred_len]

            # Handle case where TTM prediction is shorter than required
            if effective_pred_len < prediction_length:
                # Pad predictions by repeating the last predicted value
                pad_len = prediction_length - effective_pred_len
                val_preds = np.pad(
                    val_preds, ((0, 0), (0, pad_len)), mode="edge"
                )
                val_true_full = np.stack(val_targets)[:, :prediction_length]
            else:
                val_true_full = val_true
        else:
            print("    WARNING: No validation data available, falling back to point replication")
            val_preds = None

        # --- Step 2: Run inference on test set ---
        print("    Running inference on test set...")
        test_contexts, test_targets = _extract_series_from_dataset(dataset, split="test")

        test_preds = _run_ttm_inference(
            model_path=model_path,
            contexts=test_contexts,
            context_length=context_length,
            prediction_length=effective_pred_len,
            batch_size=batch_size,
            freq=freq_str,
        )

        # Handle shorter predictions
        if effective_pred_len < prediction_length:
            pad_len = prediction_length - effective_pred_len
            test_preds = np.pad(test_preds, ((0, 0), (0, pad_len)), mode="edge")

        # --- Step 3: Generate quantiles via conformal prediction ---
        print("    Generating quantile forecasts via conformal prediction...")

        if val_preds is not None and len(val_preds) > 0:
            fc_quantiles = conformal_quantiles_from_residuals(
                cal_preds=val_preds,
                cal_true=val_true_full,
                test_preds=test_preds,
                quantile_levels=quantile_levels,
            )
        else:
            # Fallback: replicate point forecast across quantiles
            fc_quantiles = np.tile(
                test_preds[:, None, :], (1, len(quantile_levels), 1)
            ).astype(np.float32)

        # --- Step 4: Save results ---
        ds_config = f"{dataset_name}/{term}"

        model_hyperparams = {
            "model_path": model_path,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "effective_prediction_length": effective_pred_len,
            "quantile_method": "conformal_prediction",
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
        print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run TTM experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["SG_Weather/D"],
        help="Dataset name(s)",
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=None,
        choices=["short", "medium", "long"],
        help="Terms to evaluate. If not specified, auto-detect from config.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ibm-granite/granite-timeseries-ttm-r2",
        help="TTM model HuggingFace path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        choices=[512, 1024, 1536],
        help="Context length (supported: 512, 1024, 1536 for R2)",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Quantile levels for conformal prediction",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to datasets.yaml config file",
    )

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
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_ttm_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_path=args.model_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=args.context_length,
                quantile_levels=args.quantiles,
                config_path=config_path,
            )
        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()