"""
PatchTST experiments for time series forecasting (via neuralforecast).

This script trains PatchTST (Nie et al., 2022) per (dataset, term, pollutant)
combination using the neuralforecast library, then produces probabilistic
forecasts on the test split. Quantile forecasts come directly from PatchTST
trained with multi-quantile loss (`MQLoss`).

Because pollutants have very different scales, a separate PatchTST model is
trained per pollutant (grouped by the trailing token of each series'
item_id, e.g. "site_ABD9_NO2" -> "NO2"). Typical AQ datasets have 6
pollutants (CO, NO2, Ozone, PM10, PM2.5, SO2), so 6 models are trained per
(dataset, term). No validation set is used -- training runs for a fixed
number of steps per pollutant group.

The entire training set is materialised into a single pandas DataFrame and
fed to neuralforecast in-memory, with `num_workers=0` so no DataLoader
subprocesses are spawned.

Usage:
    python experiments/patchtst.py
    python experiments/patchtst.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/patchtst.py --dataset "SG_Weather/D" "SG_PM25/H"
    python experiments/patchtst.py --dataset all_datasets
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
from lightning.pytorch import seed_everything
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import PatchTST

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


def _build_long_df(entries, freq: str) -> pd.DataFrame:
    """Convert GluonTS-style entries to a neuralforecast long DataFrame.

    Absolute timestamps are synthetic -- neuralforecast only needs a
    monotonic index at the correct frequency.
    """
    anchor = pd.Timestamp("2000-01-01")
    frames = []
    for i, entry in enumerate(entries):
        y = np.asarray(entry["target"], dtype=np.float32)
        ds = pd.date_range(anchor, periods=len(y), freq=freq)
        frames.append(pd.DataFrame({
            "unique_id": str(i),
            "ds": ds,
            "y": y,
        }))
    return pd.concat(frames, ignore_index=True)


def _mqloss_quantile_columns(model_alias: str, loss: MQLoss) -> list[str]:
    """Return the column names that MQLoss emits, in quantile order.

    We pull the suffixes directly from `loss.output_names` (e.g.
    "-lo-80.0", "-median", "-hi-20.0") so we stay in sync with whatever
    formatting the neuralforecast version uses.
    """
    return [f"{model_alias}{suffix}" for suffix in loss.output_names]


def run_patchtst_experiment(
    dataset_name: str,
    terms: list[str] = None,
    output_dir: str | None = None,
    context_length: int | None = None,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    encoder_layers: int = 3,
    n_heads: int = 16,
    hidden_size: int = 128,
    linear_hidden_size: int = 256,
    dropout: float = 0.2,
    patch_len: int = 16,
    stride: int = 8,
    max_steps: int = 5000,
    batch_size: int = 32,
    windows_batch_size: int = 1024,
    learning_rate: float = 1e-4,
    scaler_type: str = "standard",
):
    """
    Train one PatchTST model per pollutant and save quantile forecasts on
    the test split. No validation set is used.
    """
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = "./output/results/patchtst"
    os.makedirs(output_dir, exist_ok=True)

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'='*60}")
    print(f"Model: PatchTST (neuralforecast)")
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

        # PatchTST is univariate -- convert multivariate datasets.
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

        h = dataset.prediction_length
        # Default input_size: prediction_length (matches DeepAR default).
        # Must be a multiple of the PatchTST stride to form whole patches.
        effective_context_length = context_length if context_length is not None else h
        if effective_context_length < patch_len:
            effective_context_length = patch_len
        print(f"    - Input (context) length: {effective_context_length}")
        print(f"    - Patch len / stride: {patch_len} / {stride}")

        training_entries = list(dataset.training_dataset)
        test_inputs = list(eval_data.input)
        num_series_exp = len(training_entries)
        expected_instances = num_series_exp * num_windows
        assert len(test_inputs) == expected_instances, (
            f"Expected {expected_instances} test instances "
            f"(num_series={num_series_exp} * num_windows={num_windows}), "
            f"got {len(test_inputs)}"
        )

        # Group series by pollutant (trailing token of item_id).
        pollutant_to_indices: dict[str, list[int]] = defaultdict(list)
        for s_idx, entry in enumerate(training_entries):
            item_id = str(entry.get("item_id", s_idx))
            pollutant = item_id.rsplit("_", 1)[-1] if "_" in item_id else item_id
            pollutant_to_indices[pollutant].append(s_idx)

        pollutant_summary = {p: len(ids) for p, ids in pollutant_to_indices.items()}
        print(f"  Pollutant groups ({len(pollutant_to_indices)}): {pollutant_summary}")

        num_q = len(quantile_levels)
        fc_quantiles = np.zeros((expected_instances, num_q, h), dtype=np.float32)

        model_alias = "PatchTST"
        # Resolve MQLoss column names once (they depend on `quantile_levels`
        # but not on the pollutant).
        _loss_probe = MQLoss(quantiles=quantile_levels)
        quantile_cols = _mqloss_quantile_columns(model_alias, _loss_probe)

        pollutant_items = list(pollutant_to_indices.items())
        num_pollutants = len(pollutant_items)
        for p_idx, (pollutant, series_indices) in enumerate(pollutant_items, 1):
            print(
                f"\n  ===== [{p_idx}/{num_pollutants}] Training PatchTST for "
                f"pollutant '{pollutant}' on {len(series_indices)} series "
                f"({max_steps} steps) ====="
            )

            train_group = [training_entries[i] for i in series_indices]
            train_df = _build_long_df(train_group, dataset.freq)

            # Fresh model per pollutant to reset weights and optimizer state.
            model = PatchTST(
                h=h,
                input_size=effective_context_length,
                encoder_layers=encoder_layers,
                n_heads=n_heads,
                hidden_size=hidden_size,
                linear_hidden_size=linear_hidden_size,
                dropout=dropout,
                patch_len=patch_len,
                stride=stride,
                loss=MQLoss(quantiles=quantile_levels),
                max_steps=max_steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                windows_batch_size=windows_batch_size,
                inference_windows_batch_size=windows_batch_size,
                scaler_type=scaler_type,
                random_seed=SEED,
                val_check_steps=max_steps,  # disable validation
                start_padding_enabled=True,  # tolerate short series
                alias=model_alias,
                # Load everything in-memory -- no dataloader workers.
                dataloader_kwargs={"num_workers": 0, "persistent_workers": False},
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=False,
                enable_checkpointing=False,
                # Cap epochs so Lightning's progress bar shows a finite total
                # instead of "Epoch N/-2". `max_steps` is still the true stop
                # condition — whichever is hit first wins.
                max_epochs=max_steps,
            )

            nf = NeuralForecast(models=[model], freq=dataset.freq)
            t0 = time.perf_counter()
            nf.fit(df=train_df, val_size=0)
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

            # Build a new_df with one unique_id per test window.
            new_df = _build_long_df(group_test_inputs, dataset.freq)
            id_map = {str(i): f"w{i}" for i in range(len(group_test_inputs))}
            new_df["unique_id"] = new_df["unique_id"].map(id_map)

            print(
                f"    [{pollutant}] Predicting {len(group_test_inputs)} test windows..."
            )
            t0 = time.perf_counter()
            pred_df = nf.predict(df=new_df)
            predict_elapsed = time.perf_counter() - t0
            print(f"    [{pollutant}] prediction done in {predict_elapsed:.1f}s")

            # Place each window's forecast into the global buffer. Don't rely
            # on pred_df row order since ids are strings that sort
            # lexicographically.
            inv_id_map = {v: int(k) for k, v in id_map.items()}

            missing = [c for c in quantile_cols if c not in pred_df.columns]
            if missing:
                raise RuntimeError(
                    f"MQLoss predictions missing expected columns {missing}. "
                    f"Available: {list(pred_df.columns)}"
                )

            for uid, grp in pred_df.groupby("unique_id", sort=False):
                grp = grp.sort_values("ds")
                # (h, num_quantiles) -> (num_quantiles, h)
                q_arr = grp[quantile_cols].to_numpy().astype(np.float32).T
                original_pos = inv_id_map[str(uid)]
                dest_idx = dest_flat_indices[original_pos]
                fc_quantiles[dest_idx] = q_arr

            # Release the model before training the next pollutant.
            del nf, model

        assert fc_quantiles.shape == (expected_instances, num_q, h), (
            f"Unexpected forecast shape {fc_quantiles.shape}, "
            f"expected ({expected_instances}, {num_q}, {h})"
        )

        ds_config = f"{dataset_name}/{term}"

        model_hyperparams = {
            "model": "PatchTST",
            "library": "neuralforecast",
            "context_length": effective_context_length,
            "prediction_length": dataset.prediction_length,
            "encoder_layers": encoder_layers,
            "n_heads": n_heads,
            "hidden_size": hidden_size,
            "linear_hidden_size": linear_hidden_size,
            "dropout": dropout,
            "patch_len": patch_len,
            "stride": stride,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "windows_batch_size": windows_batch_size,
            "learning_rate": learning_rate,
            "scaler_type": scaler_type,
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
    parser = argparse.ArgumentParser(description="Run PatchTST (neuralforecast) experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Port_Activity/D"],
                        help="Dataset name(s). Single dataset, multiple datasets, or 'all_datasets'.")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Input size for PatchTST (defaults to prediction_length).")
    parser.add_argument("--config", type=str, default=None, help="Path to datasets.yaml config file")
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--linear-hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000, help="Training steps per pollutant")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--windows-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--scaler-type", type=str, default="standard",
                        choices=["identity", "standard", "robust", "minmax", "minmax1", "boxcox"])
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
            run_patchtst_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                output_dir=args.output_dir,
                context_length=args.context_length,
                config_path=config_path,
                quantile_levels=args.quantiles,
                encoder_layers=args.encoder_layers,
                n_heads=args.n_heads,
                hidden_size=args.hidden_size,
                linear_hidden_size=args.linear_hidden_size,
                dropout=args.dropout,
                patch_len=args.patch_len,
                stride=args.stride,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                windows_batch_size=args.windows_batch_size,
                learning_rate=args.learning_rate,
                scaler_type=args.scaler_type,
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
