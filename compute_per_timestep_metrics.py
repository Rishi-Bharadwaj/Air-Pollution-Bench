"""
Compute per-lead-time (hourly) metrics post-hoc from saved predictions.npz files.

Reconstructs ground truth from the raw dataset and computes metrics at each
individual forecast step h=0..pred_len-1. Output shape per metric:
    (num_series, num_windows, num_variates, pred_len)
Saved as per_timestep_metrics.npz alongside metrics.npz.

Usage:
    python compute_per_timestep_metrics.py              # process all, 20 workers
    python compute_per_timestep_metrics.py --dry-run    # test on first experiment only
    python compute_per_timestep_metrics.py --workers 8  # custom parallelism
    python compute_per_timestep_metrics.py --verify     # verify after run
    python compute_per_timestep_metrics.py --recompute  # recompute even if file exists
"""

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent
RESULTS_ROOT = REPO_ROOT / "output" / "results"
DATA_ROOT    = REPO_ROOT / "data" / "hf_dataset"
AQ_YAML      = REPO_ROOT / "aq_datasets.yaml"
HORIZONS     = ["short"]  # extend to ["short", "medium", "long"] if needed
MODELS       = {"TiRex", "visiontspp_base", "seasonal_naive", "patchtst", "dlinear"}
MAX_WORKERS  = 6

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


# ─── Dataset config ───────────────────────────────────────────────────────────
def _load_datasets_config() -> dict:
    with open(AQ_YAML) as f:
        return yaml.safe_load(f).get("datasets", {})


# ─── Discovery ────────────────────────────────────────────────────────────────
def discover_experiments() -> list[Path]:
    """Find experiment dirs with predictions.npz and config.json."""
    experiments = []
    for model_dir in sorted(RESULTS_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name not in MODELS:
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for freq_dir in sorted(ds_dir.iterdir()):
                if not freq_dir.is_dir():
                    continue
                for term_dir in sorted(freq_dir.iterdir()):
                    if not term_dir.is_dir() or term_dir.name not in HORIZONS:
                        continue
                    if not (term_dir / "predictions.npz").exists():
                        continue
                    if not (term_dir / "config.json").exists():
                        continue
                    experiments.append(term_dir)
    return experiments


# ─── Worker ───────────────────────────────────────────────────────────────────
def _worker(exp_dir_str: str, data_root_str: str, datasets_config: dict) -> tuple[str, str]:
    """
    Reconstruct ground truth and compute per-timestep metrics for one experiment.
    Saves per_timestep_metrics.npz. Returns (exp_dir_str, status_message).

    Fully vectorised — no Python loop over lead times or (series, window, variate).
    Ground truth is reconstructed by direct numpy slicing (no GluonTS iterator).

    Per-timestep RMSE = |error[h]| (RMSE of 1 value = absolute value).
    mean(pt_MSE, axis=-1) == window MSE  and  mean(pt_MAE, axis=-1) == window MAE.
    RMSE, CRPS, ND do NOT aggregate this way (different denominators).
    """
    import json
    from pathlib import Path
    import numpy as np
    import warnings
    import datasets as hf_datasets

    exp_dir   = Path(exp_dir_str)
    data_root = Path(data_root_str)

    # ── Config ────────────────────────────────────────────────────────────────
    with open(exp_dir / "config.json") as f:
        cfg = json.load(f)

    parts           = cfg["dataset_config"].split("/")  # ["AURN", "H", "short"]
    dataset_key     = f"{parts[0]}/{parts[1]}"          # "AURN/H"
    pred_len        = cfg["prediction_length"]
    num_windows     = cfg["num_windows"]
    num_series      = cfg["num_series"]
    num_variates    = cfg["num_variates"]
    seasonality     = cfg["seasonality"]
    ctx_len         = cfg["context_length"]
    quantile_levels = cfg["quantile_levels"]
    scale           = cfg.get("prediction_scale_factor", 1.0)
    test_length     = (
        datasets_config.get(dataset_key, {}).get("test_length")
        or num_windows * pred_len
    )

    # ── Predictions: (S, W, Q, V, P) ─────────────────────────────────────────
    npz_p = np.load(exp_dir / "predictions.npz")
    pq    = npz_p["predictions_quantiles"].astype(np.float64) * scale

    # ── Ground truth + context via direct numpy slicing ──────────────────────
    # Mirrors saver.py: window w covers target[test_start + w*P : test_start + (w+1)*P]
    hf_dataset = hf_datasets.load_from_disk(
        str(data_root / dataset_key)
    ).with_format("numpy")

    all_targets = hf_dataset["target"]  # list of (Ti,) arrays, one per series

    ground_truth  = np.zeros((num_series, num_windows, num_variates, pred_len))
    context_array = np.full((num_series, num_windows, num_variates, ctx_len), np.nan)

    for s, target in enumerate(all_targets):
        target     = np.asarray(target, dtype=np.float64)
        test_start = len(target) - test_length
        starts     = test_start + np.arange(num_windows) * pred_len  # (W,)

        # Ground truth: vectorised index array
        idx = starts[:, None] + np.arange(pred_len)[None, :]         # (W, P)
        ground_truth[s, :, 0, :] = target[idx]

        # Context: variable start, minimal loop
        for w, fs in enumerate(starts.tolist()):
            cs  = max(0, fs - ctx_len)
            ctx = target[cs:fs]
            context_array[s, w, 0, :len(ctx)] = ctx

    # ── Fully vectorised metric computation ───────────────────────────────────
    gt          = ground_truth                              # (S, W, V, P)
    median_idx  = list(quantile_levels).index(0.5)
    median_pred = pq[:, :, median_idx, :, :]               # (S, W, V, P)
    error       = gt - median_pred
    abs_error   = np.abs(error)
    abs_gt      = np.abs(gt)

    # MSE / MAE / RMSE
    pt_mse  = error ** 2
    pt_mae  = abs_error
    pt_rmse = abs_error   # RMSE[single value] = |value|

    # MAPE / sMAPE
    with np.errstate(divide="ignore", invalid="ignore"):
        pt_mape  = abs_error / abs_gt
        pt_mape  = np.where(np.isfinite(pt_mape), pt_mape, np.nan)
        pt_smape = 2 * abs_error / (abs_gt + np.abs(median_pred))
        pt_smape = np.where(np.isfinite(pt_smape), pt_smape, np.nan)

    # MASE: seasonal_error[s,w,v] from context — vectorised via nanmean on diff
    # NaN padding is at the END of context_array, so seasonal diff with any NaN → NaN,
    # and nanmean skips those positions automatically.
    seasonal_diff = np.abs(
        context_array[..., seasonality:] - context_array[..., :-seasonality]
    )                                                       # (S, W, V, ctx_len-seasonality)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # suppress all-NaN slice warning
        se = np.nanmean(seasonal_diff, axis=-1)             # (S, W, V)
    se = np.where(se > 0, se, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pt_mase = abs_error / se[:, :, :, np.newaxis]      # broadcast over P

    # ND per timestep = |error[h]| / |gt[h]|
    with np.errstate(divide="ignore", invalid="ignore"):
        pt_nd = np.where(abs_gt > 0, abs_error / abs_gt, np.nan)

    # CRPS: accumulate weighted quantile losses over Q quantiles
    pt_crps = np.zeros_like(gt)
    for q_idx, q in enumerate(quantile_levels):
        q_pred  = pq[:, :, q_idx, :, :]                    # (S, W, V, P)
        q_error = gt - q_pred
        ind     = (q_pred >= gt).astype(np.float64)
        q_loss  = 2.0 * np.abs(q_error * (ind - q))
        with np.errstate(divide="ignore", invalid="ignore"):
            pt_crps += np.where(abs_gt > 0, q_loss / abs_gt, 0.0)
    pt_crps /= len(quantile_levels)
    pt_crps   = np.where(abs_gt > 0, pt_crps, np.nan)

    result = {
        "MSE":   pt_mse,
        "MAE":   pt_mae,
        "RMSE":  pt_rmse,
        "MAPE":  pt_mape,
        "sMAPE": pt_smape,
        "MASE":  pt_mase,
        "ND":    pt_nd,
        "CRPS":  pt_crps,
    }

    np.savez_compressed(exp_dir / "per_timestep_metrics.npz", **result)
    return exp_dir_str, f"OK — shape {pt_mse.shape}"


# ─── Verify ───────────────────────────────────────────────────────────────────
def verify_all() -> None:
    """Check MSE/MAE consistency: mean(per_timestep, axis=-1) should equal window metric."""
    rows = []
    for exp_dir in discover_experiments():
        pt_path = exp_dir / "per_timestep_metrics.npz"
        m_path  = exp_dir / "metrics.npz"
        if not pt_path.exists() or not m_path.exists():
            continue
        pt = np.load(pt_path)
        m  = np.load(m_path)
        mse_diff = float(np.nanmax(np.abs(np.nanmean(pt["MSE"], axis=-1) - m["MSE"])))
        mae_diff = float(np.nanmax(np.abs(np.nanmean(pt["MAE"], axis=-1) - m["MAE"])))
        rel = exp_dir.relative_to(RESULTS_ROOT).parts
        rows.append((rel[0], f"{rel[1]}/{rel[2]}", rel[3], str(pt["MSE"].shape), mse_diff, mae_diff))

    if not rows:
        print("No experiments found with per_timestep_metrics.npz.")
        return

    print(f"{'model':<24}  {'dataset':<14}  {'term':<6}  {'shape':<24}  {'mse_diff':>10}  {'mae_diff':>10}")
    print("-" * 100)
    for model, ds, term, shape, mse_d, mae_d in rows:
        print(f"{model:<24}  {ds:<14}  {term:<6}  {shape:<24}  {mse_d:>10.2e}  {mae_d:>10.2e}")
    max_mse = max(r[4] for r in rows)
    max_mae = max(r[5] for r in rows)
    print(f"\nMax MSE diff: {max_mse:.2e}  |  Max MAE diff: {max_mae:.2e}  (both should be ~0)")


# ─── Debug Verify ─────────────────────────────────────────────────────────────
def debug_verify_all() -> None:
    """
    Extended verification: for each experiment find the worst (s, w, v) and compare
    the actual mae_diff against the theoretical float16 quantisation contribution.
    Uses only already-saved .npz and config.json files — no HF dataset access needed.
    """
    header = (
        f"{'model':<24}  {'dataset':<14}  {'term':<6}  {'scale':>6}  "
        f"{'max_pred':>10}  {'exp_f16_mae':>12}  {'mae_diff':>10}  {'item_id'}"
    )
    print(header)
    print("-" * len(header))

    global_worst = None  # (mae_diff, exp_dir, worst_s, worst_w, worst_v)

    for exp_dir in discover_experiments():
        pt_path   = exp_dir / "per_timestep_metrics.npz"
        m_path    = exp_dir / "metrics.npz"
        pred_path = exp_dir / "predictions.npz"
        cfg_path  = exp_dir / "config.json"
        if not (pt_path.exists() and m_path.exists() and pred_path.exists() and cfg_path.exists()):
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)

        scale           = cfg.get("prediction_scale_factor", 1.0)
        item_ids        = cfg.get("item_ids", [])
        quantile_levels = cfg.get("quantile_levels", [])
        median_idx      = quantile_levels.index(0.5) if 0.5 in quantile_levels else 4

        pt = np.load(pt_path)
        m  = np.load(m_path)
        pq = np.load(pred_path)["predictions_quantiles"]  # float16, (S, W, Q, V, P)

        mae_diff_arr = np.abs(np.nanmean(pt["MAE"], axis=-1) - m["MAE"])  # (S, W, V)
        flat_idx     = int(np.nanargmax(mae_diff_arr))
        worst_s, worst_w, worst_v = np.unravel_index(flat_idx, mae_diff_arr.shape)
        worst_mae_diff = float(mae_diff_arr[worst_s, worst_w, worst_v])

        # Float16 quantisation contribution at the worst (s, w, v)
        pred_f16 = pq[worst_s, worst_w, median_idx, worst_v, :].astype(np.float64) * scale
        # ULP-based estimate: for each already-quantised float16 value, the next
        # representable float16 gives the step size; original float64 pred lies within
        # ±ULP/2 of the stored value.  Expected |error| per step ~ ULP/2.
        ulp = np.array([
            abs(float(np.nextafter(np.float16(v / scale), np.float16(np.inf)))
                - float(np.float16(v / scale))) * scale
            if np.isfinite(np.float16(v / scale)) else np.nan
            for v in pred_f16
        ])
        exp_f16_mae = float(np.nanmean(ulp / 2))
        max_pred    = float(np.abs(pred_f16).max())

        iid = item_ids[worst_s] if worst_s < len(item_ids) else f"s={worst_s}"
        rel = exp_dir.relative_to(RESULTS_ROOT).parts
        model, ds, term = rel[0], f"{rel[1]}/{rel[2]}", rel[3]

        print(
            f"{model:<24}  {ds:<14}  {term:<6}  {scale:>6.0f}  "
            f"{max_pred:>10.1f}  {exp_f16_mae:>12.4f}  {worst_mae_diff:>10.4f}  {iid}"
        )

        if global_worst is None or worst_mae_diff > global_worst[0]:
            global_worst = (worst_mae_diff, exp_dir, worst_s, worst_w, worst_v,
                            cfg, pq, pt, m, exp_f16_mae, max_pred, scale, median_idx)

    if global_worst is None:
        print("No experiments found.")
        return

    (worst_mae_diff, exp_dir, worst_s, worst_w, worst_v,
     cfg, pq, pt, m, exp_f16_mae, max_pred, scale, median_idx) = global_worst

    item_ids        = cfg.get("item_ids", [])
    quantile_levels = cfg.get("quantile_levels", [])
    iid             = item_ids[worst_s] if worst_s < len(item_ids) else f"s={worst_s}"

    print(f"\n{'─'*70}")
    print(f"Drill-down: {exp_dir.relative_to(RESULTS_ROOT)}")
    print(f"  worst (s={worst_s}, w={worst_w}, v={worst_v})  item_id: {iid}")
    print(f"  scale_factor         : {scale}")
    print(f"  max |pred| at (s,w,v): {max_pred:.2f}")
    print(f"  expected f16 MAE     : {exp_f16_mae:.4f}  (mean round-trip error per step)")
    print(f"  actual   mae_diff    : {worst_mae_diff:.4f}")
    ratio = worst_mae_diff / exp_f16_mae if exp_f16_mae > 0 else float("inf")
    print(f"  ratio actual/f16     : {ratio:.2f}  (≈1 → float16; >>1 → other cause)")
    print(f"  win_MAE  (metrics.npz)  : {float(m['MAE'][worst_s, worst_w, worst_v]):.4f}")
    print(f"  pt_MAE   (pt_npz mean)  : {float(np.nanmean(pt['MAE'][worst_s, worst_w, worst_v])):.4f}")

    # Show first few lead-time values of the prediction for the worst window
    pred_f16 = pq[worst_s, worst_w, median_idx, worst_v, :].astype(np.float64) * scale
    ulp_worst = np.array([
        abs(float(np.nextafter(np.float16(v / scale), np.float16(np.inf)))
            - float(np.float16(v / scale))) * scale
        if np.isfinite(np.float16(v / scale)) else np.nan
        for v in pred_f16
    ])
    print(f"\n  pred (float16→f64) first 6 steps: {pred_f16[:6]}")
    print(f"  ULP/2 (max f16 err) first 6 steps: {ulp_worst[:6] / 2}")
    print(f"  pt_MAE per step     first 6 steps: {pt['MAE'][worst_s, worst_w, worst_v, :6]}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run",      action="store_true", help="Process only the first experiment to test")
    parser.add_argument("--verify",       action="store_true", help="Run verification table after processing")
    parser.add_argument("--debug-verify", action="store_true", help="Extended verification: diagnose float16 vs other causes")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Parallel workers (default: {MAX_WORKERS})")
    args = parser.parse_args()

    datasets_config = _load_datasets_config()
    # experiments = discover_experiments()
    
    # print(f"Experiments to process: {len(experiments)}")
    experiments=None
    if not experiments:
        print("Nothing to do.")
        if args.verify:
            verify_all()
        if args.debug_verify:
            debug_verify_all()
        return

    if args.dry_run:
        experiments = experiments[:1]
        print(f"Dry-run: processing {experiments[0].relative_to(RESULTS_ROOT)}")

    # Single-process dry-run; parallel otherwise
    if args.dry_run or len(experiments) == 1:
        exp_dir_str, status = _worker(str(experiments[0]), str(DATA_ROOT), datasets_config)
        print(f"  {Path(exp_dir_str).relative_to(RESULTS_ROOT)}: {status}")

        # Quick sanity check
        pt = np.load(experiments[0] / "per_timestep_metrics.npz")
        m  = np.load(experiments[0] / "metrics.npz")
        mse_diff = float(np.nanmax(np.abs(np.nanmean(pt["MSE"], axis=-1) - m["MSE"])))
        mae_diff = float(np.nanmax(np.abs(np.nanmean(pt["MAE"], axis=-1) - m["MAE"])))
        print(f"\nSanity: max |mean(pt_MSE) - win_MSE| = {mse_diff:.2e}  (float16 rounding, MAE diff is more diagnostic)")
        print(f"Sanity: max |mean(pt_MAE) - win_MAE| = {mae_diff:.2e}  (should be <0.1; float16 quantisation only)")

        rmse = np.nanmean(pt["RMSE"], axis=(0, 1, 2))
        print(f"\nRMSE by lead time (mean over all sites/windows):")
        for h in [0, len(rmse)//4, len(rmse)//2, len(rmse)-1]:
            print(f"  h={h:<3}: {rmse[h]:.4f}")
    else:
        successes, failures = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_worker, str(d), str(DATA_ROOT), datasets_config): d
                for d in experiments
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                d = futures[future]
                try:
                    _, status = future.result()
                    successes.append(d)
                    print(f"[{done}/{len(experiments)}] OK  {d.relative_to(RESULTS_ROOT)}")
                except Exception as exc:
                    failures.append((d, exc))
                    print(f"[{done}/{len(experiments)}] ERR {d.relative_to(RESULTS_ROOT)}: {exc}")

        print(f"\nDone: {len(successes)} succeeded, {len(failures)} failed.")
        if failures:
            print("Failed:")
            for d, e in failures:
                print(f"  {d.relative_to(RESULTS_ROOT)}: {e}")

    if args.verify:
        print()
        verify_all()

    if args.debug_verify:
        print()
        debug_verify_all()


if __name__ == "__main__":
    main()
