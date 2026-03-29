"""
Plot actual vs predicted for a given item_id and forecast window.

Usage:
    python plot_forecast.py --item-id "site_105_North_Campus_DU_Delhi_IMD_CO" --window 0
    python plot_forecast.py --item-id "site_108_Aya_Nagar_Delhi_IMD_PM2.5" --window 5 --context 48
    python plot_forecast.py --item-id "site_108_Aya_Nagar_Delhi_IMD_PM2.5" --window 5 --save
"""

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk


def plot_forecast(
    results_dir: str,
    hf_dataset: str,
    item_id: str,
    window_idx: int,
    context_steps: int = 72,
    save: bool = False,
):
    # ── Load predictions & config ─────────────────────────────────────────────
    data = np.load(Path(results_dir) / "predictions.npz")
    with open(Path(results_dir) / "config.json") as f:
        cfg = json.load(f)

    item_ids = cfg["item_ids"]
    if item_id not in item_ids:
        raise ValueError(
            f"item_id '{item_id}' not found. Available:\n  " + "\n  ".join(item_ids)
        )
    series_idx = item_ids.index(item_id)

    num_windows = cfg["num_windows"]
    if not (0 <= window_idx < num_windows):
        raise ValueError(f"window_idx {window_idx} out of range [0, {num_windows - 1}]")

    preds = data["predictions_quantiles"].astype(float) * cfg["prediction_scale_factor"]
    ql    = np.array(cfg["quantile_levels"])
    freq  = cfg["freq"]

    timestamps = pd.to_datetime(data["timestamps"][series_idx, window_idx], unit="s")

    q10_idx = np.argmin(np.abs(ql - 0.1))
    q50_idx = np.argmin(np.abs(ql - 0.5))
    q90_idx = np.argmin(np.abs(ql - 0.9))

    pred_q10    = preds[series_idx, window_idx, q10_idx, 0, :]
    pred_median = preds[series_idx, window_idx, q50_idx, 0, :]
    pred_q90    = preds[series_idx, window_idx, q90_idx, 0, :]

    # ── Build actual series from HF dataset ───────────────────────────────────
    ds   = load_from_disk(hf_dataset)
    item = ds[series_idx]

    target = np.array(item["target"])
    if target.ndim == 2:
        target = target[0]

    start_raw    = item["start"]
    series_start = pd.Timestamp(start_raw.item() if hasattr(start_raw, "item") else start_raw)
    freq_offset  = pd.tseries.frequencies.to_offset(freq)
    series_index = pd.date_range(series_start, periods=len(target), freq=freq_offset)
    series       = pd.Series(target, index=series_index)

    # ── Slice actual values (one solid range: context + forecast window) ───────
    forecast_start = timestamps[0]
    context_start  = forecast_start - context_steps * freq_offset
    actual_all     = series[context_start : timestamps[-1]]
    actual_forecast = series.reindex(timestamps)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))

    # Single solid actual line — no gap
    ax.plot(actual_all.index, actual_all.values, color="steelblue", lw=1.5, label="Actual")
    ax.scatter(actual_forecast.index, actual_forecast.values, color="steelblue", s=12, zorder=5, label="Actual (forecast window)")
    ax.plot(timestamps, pred_median, color="tomato", lw=1.5, label="Predicted (median)")
    ax.fill_between(timestamps, pred_q10, pred_q90, color="tomato", alpha=0.2, label="10th–90th pct")
    ax.axvline(forecast_start, color="gray", linestyle=":", alpha=0.8, label="Forecast start")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper left")
    ax.set_title(f"{item_id}  |  Window {window_idx}")
    ax.set_ylabel("Value")
    plt.tight_layout()

    if save:
        out_path = Path(results_dir) / f"plot_{item_id}_w{window_idx}.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot actual vs predicted forecast")
    parser.add_argument("--results-dir", type=str,
                        default="./output/results/chronos_bolt_base/CPCB/H/short")
    parser.add_argument("--hf-dataset", type=str,
                        default="./data/hf_dataset/CPCB/H/")
    parser.add_argument("--item-id", type=str, required=True,
                        help="item_id of the series to plot")
    parser.add_argument("--window", type=int, default=0,
                        help="Forecast window index")
    parser.add_argument("--context", type=int, default=72,
                        help="Number of historical steps to show before the forecast")
    parser.add_argument("--save", action="store_true",
                        help="Save plot to file instead of displaying")
    args = parser.parse_args()

    plot_forecast(
        results_dir=args.results_dir,
        hf_dataset=args.hf_dataset,
        item_id=args.item_id,
        window_idx=args.window,
        context_steps=args.context,
        save=args.save,
    )


if __name__ == "__main__":
    main()