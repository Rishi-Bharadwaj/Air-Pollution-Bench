# AtmoBench

**AtmoBench** is a large-scale multi-country, multi-pollutant dataset and benchmark for short-term air quality forecasting using Time Series Foundation Models (TSFMs). It covers 6 major pollutants over a three-year period across 7 countries and 4 continents, with more than 14,000 station-pollutant series, providing a standardised, reproducible protocol for evaluating TSFMs in a zero-shot setting.

---

## Background

Air pollution causes an estimated 7.9 million premature deaths annually, making accurate forecasting a critical public health priority. Machine learning is increasingly being applied to forecast air pollution levels, yet existing benchmarks remain narrow in both geographic scope and pollutant coverage, and fail to evaluate the latest generation of time series foundation models (TSFMs) on real-world, large-scale data. We present AtmoBench, a large-scale multi-country and multi-pollutant dataset and benchmark to address this gap. AtmoBench covers 6 major pollutants over a three-year period across 7 diverse countries and 4 continents, with more than 14,000 station-pollutant series, aiming to provide a comprehensive benchmark for air quality tasks. We benchmark this dataset across 11 leading time series foundation models and classical baselines to assess performance on short-term air quality forecasting. Our results demonstrate that TSFMs are effective zero-shot forecasters and consistently outperform classical baselines, with our top-performing model employing a cross-modal architecture that leverages a vision foundation model for time series forecasting.

---

## Dataset

Data is sourced from seven national and regional air quality monitoring networks:

| Network | Country / Region | Agency |
|---------|-----------------|--------|
| EPA (AQS) | United States | Environmental Protection Agency |
| CPCB | India | Central Pollution Control Board |
| CNEMC | China | China National Environmental Monitoring Centre |
| AURN | United Kingdom | Automatic Urban and Rural Network |
| EEA (France) | France | European Environment Agency |
| EEA (Germany) | Germany | European Environment Agency |
| SINAICA | Mexico | Sistema Nacional de Información de la Calidad del Aire |

**Pollutants:** PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃

**Scale:** 14,000+ station-pollutant series across 7 countries and 4 continents

**Time range:** July 2022 – June 2025 (3 years, hourly resolution)

**Evaluation protocol:** 168-hour (1-week) context window, 24-hour prediction horizon, evaluated over a 2-year rolling test period.

**Imputation:** Stations with more than 30% missing data are excluded. Gaps of up to 336 hours (2 weeks) are filled by linear interpolation; longer gaps are left as-is and handled by the evaluation harness.

---

## Models Evaluated

**TSFMs (zero-shot):** Chronos-Bolt, Chronos 2, Moirai, Moirai 2, TimesFM 1.0 / 2.0 / 2.5, TiRex, VisionTS++, Sundial, Kairos

**Supervised baselines:** DLinear, PatchTST (trained per-dataset)

**Classical baselines:** Seasonal Naïve, AutoARIMA, AutoETS

**Metrics:** MASE (point accuracy) and CRPS (probabilistic calibration), both normalised by Seasonal Naïve.


---

## Repository Structure

```
AtmoBench/
├── pyproject.toml              # unified dependencies for the data pipeline
│
├── data/
│   ├── data.yaml               # paths and config for all data pipeline stages
│   ├── data.md                 # data pipeline documentation
│   ├── data_download_scripts/  # notebooks/scripts to generate download URLs
│   │   ├── epa_download.ipynb
│   │   ├── cpcb_download.ipynb
│   │   ├── cnemc_download.ipynb
│   │   ├── aurn_download.ipynb
│   │   └── openaq_download.ipynb
│   ├── data_preprocess_scripts/ # per-network preprocessing (raw → cleaned CSVs)
│   │   ├── epa_preprocess.py
│   │   ├── cpcb_preprocess.py
│   │   ├── cnemc_preprocess.py
│   │   ├── aurn_preprocess.py
│   │   ├── eea_preprocess.py
│   │   ├── sinaica_preprocess.py  # also handles download via Rsinaica API
│   │   └── openaq_preprocess.py
│   ├── imputation.py           # filters stations and fills short gaps
│   ├── visualise.py            # heatmap visualisation of coverage and missingness
│   └── aq_dataset_builder.py  # converts imputed CSVs → HuggingFace Arrow datasets
│
└── eval/
    ├── config.yaml             # model list, dataset paths, leaderboard settings
    ├── datasets.yaml           # test/val lengths and prediction horizon per dataset
    ├── eval.md                 # evaluation documentation
    ├── run.py                  # main evaluation runner
    ├── compute_local_leaderboard.py  # aggregates results into ranked tables
    ├── leaderboard_helpers.py  # normalisation and consistency checks
    ├── leaderboard_utils.py    # display utilities, model group definitions
    ├── data_statistics.ipynb   # dataset statistics and analysis
    ├── plot_forecast.ipynb     # forecast visualisation
    ├── src/timebench/          # evaluation harness (adapted from TIME framework)
    │   └── evaluation/
    │       ├── data.py
    │       ├── dataset_builder.py
    │       ├── metrics.py
    │       ├── saver.py
    │       └── utils.py
    └── experiments/            # one script per model
        ├── chronos_bolt.py
        ├── chronos2.py
        ├── moirai.py
        ├── moirai2.py
        ├── timesfm1.0.py
        ├── timesfm2.0.py
        ├── timesfm2.5.py
        ├── tirex_model.py
        ├── visiontspp.py
        ├── sundial.py
        ├── kairos_model.py
        ├── dlinear.py
        ├── patchtst.py
        ├── lgbm.py
        ├── deepar.py
        ├── seasonal_naive.py
        └── auto_ets.py
```

The evaluation harness is adapted from the [TIME framework](https://github.com/zqiao11/TIME).

---

## Downloading the Datasets

The preprocessed and imputed AtmoBench datasets are available on Zenodo:

> [https://doi.org/10.5281/zenodo.19643640](https://doi.org/10.5281/zenodo.19643640)

This includes the HuggingFace Arrow datasets (one per network) ready for use with the evaluation harness. If you prefer to reproduce the datasets from scratch, follow the Data Pipeline steps below.

---

## Data Pipeline

Each network goes through the same four stages, all configured via `data/data.yaml`:

1. **Download** — download scripts generate lists of URLs (or use an API scraper for SINAICA) which are then fetched with `aria2c`. EEA data can be bulk-downloaded directly from the EEA data hub. SINAICA has one combioned script for scarping and preprocessing.

2. **Preprocess** — per-network scripts convert raw files to a unified format: one CSV per station per pollutant, all timestamps converted to local time with unit normalisation applied.

3. **Visualise** — `visualise.py` produces heatmaps showing data coverage over time for each station, useful for inspecting gaps before imputation.

4. **Impute** — `imputation.py` drops stations below the 70% completeness threshold and fills remaining gaps (up to 336 hours) via linear interpolation.

After imputation, `aq_dataset_builder.py` converts the per-station CSVs into HuggingFace Arrow datasets (one per network) ready for the evaluation harness.

---

## Evaluation

The evaluation harness is driven by `eval/config.yaml`. It is adapted from the TIME leaderboard, (https://github.com/zqiao11/TIME). To run evaluations:

```bash
cd eval
python run.py
```

Models, datasets, and hyperparameters are all declared in `config.yaml`. Each experiment script in `experiments/` is self-contained and can also be run directly.

To compute the leaderboard from saved results:

```bash
cd eval
python compute_local_leaderboard.py
```

Results are aggregated across datasets and normalised by Seasonal Naïve, with separate per-pollutant breakdowns.

---

## Setup

```bash
# Create environment and install data pipeline dependencies
uv venv
uv pip install -e .

```

Most eval dependencies (Chronos, Moirai, TimesFM, etc.) are installed on-demand by `run.py` per model, as declared in `config.yaml`.

---

## Citation

If you use AtmoBench in your work, please cite:

```bibtex
@inproceedings{
bharadwaj2026air,
title={Air Quality Arena: A Large-Scale Multi-Region Ground Monitoring Dataset and Benchmark for Air Quality Forecasting with Time-Series Foundation Models},
author={Rishi Bharadwaj and Manik Gupta and Pandarasamy Arjunan},
booktitle={2nd ICML Workshop on Foundation Models for Structured Data},
year={2026},
url={https://openreview.net/forum?id=CUXaSOR9fA}
}

```

Please also cite the [TIME framework](https://github.com/zqiao11/TIME) which the evaluation harness is based on.

---

## License

Dataset redistribution is subject to the terms of each source network. See the readme included with our datasets on zenodo for details.