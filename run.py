"""
TIME benchmark runner.

Usage:
    python run.py --config config.yaml
    python run.py --config config.yaml model=chronos_bolt
    python run.py --config config.yaml dataset=SG_PM25/H
    python run.py --config config.yaml model=chronos_bolt dataset=Water_Quality_Darwin
    python run.py --config config.yaml model=chronos_bolt dataset=Water_Quality_Darwin/15T
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from packaging.version import Version
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_overrides(overrides: list[str]) -> dict:
    result = {}
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"Invalid override '{o}', expected key=value")
        k, v = o.split("=", 1)
        result[k] = v
    return result


def resolve_datasets(dataset_override: str | None, config: dict, datasets_config: dict) -> list[str]:
    all_datasets = list(datasets_config.get("datasets", {}).keys())

    if dataset_override is None:
        return config.get("datasets", all_datasets)

    # Exact match (e.g. "Water_Quality_Darwin/15T")
    if dataset_override in datasets_config.get("datasets", {}):
        return [dataset_override]

    # Prefix match (e.g. "Water_Quality_Darwin" -> all freqs)
    matches = [d for d in all_datasets if d.startswith(dataset_override + "/")]
    if matches:
        return matches

    raise ValueError(
        f"Dataset '{dataset_override}' not found in datasets config.\n"
        f"Available: {all_datasets}"
    )


def run_experiment(
    model_name: str,
    model_cfg: dict,
    dataset: str,
    data_dir: Path,
    datasets_config_path: Path,
    time_repo: Path,
) -> int:
    script = time_repo / model_cfg["script"]
    packages = model_cfg.get("packages", [])
    extra_args = model_cfg.get("args", {})
    git_clone = model_cfg.get("git_clone")

    if git_clone:
        clone_dest = time_repo / git_clone["dest"]
        if not (clone_dest / ".git").exists():
            print(f"Cloning {git_clone['url']} -> {clone_dest}")
            clone_dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", git_clone["url"], str(clone_dest)], check=True)

    # by uv's ephemeral --with environments.
    venv_python = time_repo / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = time_repo / ".venv" / "bin" / "python"
    python_bin = str(venv_python) if venv_python.exists() else "python"

    if packages:
        install_cmd = ["uv", "pip", "install"] + packages
        print(f"Installing packages: {' '.join(packages)}")
        subprocess.run(install_cmd, cwd=time_repo, check=True)

        # Only reinstall torch from CUDA index if it's missing or below the required version
        torch_check = subprocess.run(
            [python_bin, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True,
        )
        needs_torch = torch_check.returncode != 0
        if not needs_torch:
            
            installed = torch_check.stdout.strip().split("+")[0]  # strip +cu128 suffix
            needs_torch = Version(installed) != Version("2.10.0")

        if needs_torch:
            print(f"Installing torch==2.10.0 (current: {torch_check.stdout.strip() if torch_check.returncode == 0 else 'not found'})")
            subprocess.run(
                ["uv", "pip", "install", "torch==2.10.0",
                 "--index-url", "https://download.pytorch.org/whl/cu128"],
                cwd=time_repo, check=True,
            )
        else:
            print(f"torch {torch_check.stdout.strip()} already installed, skipping reinstall")

    cmd = [
        python_bin, str(script),
        "--dataset", dataset,
        "--config", str(datasets_config_path),
    ]
    for k, v in extra_args.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]

    env = os.environ.copy()
    env["TIME_DATASET"] = str(data_dir)
    env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::DeprecationWarning"

    print(f"\n{'='*60}")
    print(f"Model:         {model_name}")
    print(f"Dataset:       {dataset}")
    print(f"TIME_DATASET:  {data_dir}")
    print(f"Command:       {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env, cwd=time_repo)
    if result.returncode != 0:
        print(f"ERROR: Failed for model={model_name} dataset={dataset} (exit {result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="TIME benchmark runner")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Overrides as key=value pairs (e.g. model=chronos_bolt dataset=SG_PM25)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_root = config_path.parent
    config = load_config(config_path)
    overrides = parse_overrides(args.overrides)

    # Resolve paths relative to the config file's directory
    time_repo = (config_root / config["time_repo"]).resolve()
    data_dir = (config_root / config["data_dir"]).resolve()
    datasets_config_path = (config_root / config["datasets_config"]).resolve()

    with open(datasets_config_path) as f:
        datasets_config = yaml.safe_load(f)

    # Resolve model(s) to run
    model_override = overrides.get("model")
    all_models = config["models"]
    if model_override and model_override != "all":
        if model_override not in all_models:
            raise ValueError(
                f"Model '{model_override}' not in config. Available: {list(all_models)}"
            )
        models_to_run = {model_override: all_models[model_override]}
    else:
        models_to_run = all_models

    # Resolve dataset(s) to run
    dataset_override = overrides.get("dataset")
    datasets_to_run = resolve_datasets(dataset_override, config, datasets_config)

    print(f"Models:   {list(models_to_run)}")
    print(f"Datasets: {datasets_to_run}")

    failures = []
    for model_name, model_cfg in models_to_run.items():
        for dataset in datasets_to_run:
            rc = run_experiment(
                model_name, model_cfg, dataset, data_dir, datasets_config_path, time_repo
            )
            if rc != 0:
                failures.append((model_name, dataset))

    print(f"\n{'='*60}")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for model, ds in failures:
            print(f"  {model} / {ds}")
        sys.exit(1)
    else:
        print("All experiments completed successfully.")


if __name__ == "__main__":
    main()
