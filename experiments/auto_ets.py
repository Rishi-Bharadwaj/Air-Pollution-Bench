"""
AutoETS baseline experiments for time series forecasting.

This script uses the AutoETS model from statsforecast. AutoETS automatically
selects the best Error/Trend/Seasonal exponential-smoothing model for each
series. It is typically much faster than AutoARIMA on short series with
strong seasonality.

Usage:
    python experiments/auto_ets.py
    python experiments/auto_ets.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/auto_ets.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/auto_ets.py --dataset all_datasets  # Run all datasets from config
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from tqdm.auto import tqdm

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms

# Load environment variables
load_dotenv()


def run_auto_ets_experiment(
    dataset_name: str,
    terms: list[str] = None,
    output_dir: str | None = None,
    context_length: int | None = None,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    n_jobs: int = 1,
):
    """
    Run AutoETS baseline experiments on a dataset with specified terms.

    Args:
        dataset_name: Dataset name (e.g., "SG_Weather/D")
        terms: List of terms to evaluate ("short", "medium", "long")
        output_dir: Output directory for results
        context_length: Maximum context length; crops to last N timesteps if longer
        config_path: Path to datasets.yaml config file
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
        output_dir = "./output/results/auto_ets"

    os.makedirs(output_dir, exist_ok=True)

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


    print(f"\n{'='*60}")
    print(f"Model: AutoETS")
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

        # Initialize the dataset
        to_univariate = False if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim == 1 else True

        # Load dataset with config settings
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        season_length = get_seasonality(dataset.freq)

        data_length = test_length
        num_windows = dataset.windows
        split_name = "Test split"
        eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")
        print(f"    - Season length: {season_length}")

        # Generate predictions
        orig_inputs = eval_data.input
        if context_length is not None:
            inputs = []
            for d in orig_inputs:
                new_d = d.copy()
                new_d["target"] = d["target"][-context_length:]
                inputs.append(new_d)
        else:
            inputs = list(orig_inputs)

        # Build one long-format DataFrame: one unique_id per test window.
        # ds values are synthetic -- statsforecast only needs a monotonic
        # datetime index at the correct frequency; absolute dates do not
        # affect the AutoETS fit.
        anchor = pd.Timestamp("2000-01-01")
        frames = []
        for i, entry in enumerate(tqdm(inputs, desc=f"AutoETS {term} (prep)")):
            y = np.asarray(entry["target"], dtype=float)
            ds = pd.date_range(anchor, periods=len(y), freq=dataset.freq)
            frames.append(pd.DataFrame({
                "unique_id": i,
                "ds": ds,
                "y": y,
            }))
        long_df = pd.concat(frames, ignore_index=True)

        sf = StatsForecast(
            models=[AutoETS(season_length=season_length, model="AZA")],
            freq=dataset.freq,
            n_jobs=n_jobs,
            verbose=True
        )
        print(f"  Fitting AutoETS on {len(inputs)} windows with n_jobs={n_jobs}...")
        fc_df = sf.forecast(
            df=long_df,
            h=dataset.prediction_length,
            level=[20, 40, 60, 80],
        )
        # Check for exploding intervals
        # Map requested quantiles to statsforecast prediction-interval columns
        # (symmetric Gaussian intervals; mean == median for ETS additive models).
        q_cols = [
            "AutoETS-lo-80",  # 0.1
            "AutoETS-lo-60",  # 0.2
            "AutoETS-lo-40",  # 0.3
            "AutoETS-lo-20",  # 0.4
            "AutoETS",        # 0.5
            "AutoETS-hi-20",  # 0.6
            "AutoETS-hi-40",  # 0.7
            "AutoETS-hi-60",  # 0.8
            "AutoETS-hi-80",  # 0.9
        ]
        fc_df = fc_df.sort_values(["unique_id", "ds"])
        
        arr = fc_df[q_cols].to_numpy()  # (num_instances * h, num_quantiles)
        num_instances = len(inputs)
        h = dataset.prediction_length
        # (num_instances, h, num_quantiles) -> (num_instances, num_quantiles, h)
        assert len(fc_df) == num_instances * h, f"Expected {num_instances * h} rows, got {len(fc_df)}"
        fc_quantiles = arr.reshape(num_instances, h, len(q_cols)).transpose(0, 2, 1)

        # Compute metrics
        ds_config = f"{dataset_name}/{term}"

        # Prepare model hyperparameters for metadata
        model_hyperparams = {
            "model": "AutoETS",
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
        )
        print(f"  Completed: {metadata['num_series']} series x {metadata['num_windows']} windows")
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run AutoETS baseline experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Port_Activity/D"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets' to run all datasets from config")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Maximum context length; crops to last N timesteps if series is longer")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel workers for per-instance ETS fits (-1 = all cores)")
    args = parser.parse_args()

    # Handle dataset list or 'all_datasets'
    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        # Load all datasets from config
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    # Iterate over all datasets
    total_datasets = len(datasets)
    failed = False
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_auto_ets_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                output_dir=args.output_dir,
                context_length=args.context_length,
                config_path=config_path,
                n_jobs=args.n_jobs,
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
