"""Main orchestration script for C3-AutoCoT experiments."""

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Orchestrate a single run.
    
    This script:
    1. Applies mode-specific overrides (sanity_check, pilot, main)
    2. Invokes train.py as a subprocess
    """
    
    # Determine mode
    mode = None
    if "--sanity_check" in sys.argv:
        mode = "sanity_check"
        sys.argv.remove("--sanity_check")
    elif "--pilot" in sys.argv:
        mode = "pilot"
        sys.argv.remove("--pilot")
    elif "--main" in sys.argv:
        mode = "main"
        sys.argv.remove("--main")
    else:
        print("Error: Must specify one of: --sanity_check, --pilot, --main")
        sys.exit(1)
        
    print(f"Mode: {mode}")
    
    # Apply mode-specific overrides
    if mode == "sanity_check":
        OmegaConf.update(cfg, "mode.sanity_check", True)
        OmegaConf.update(cfg, "wandb.mode", "disabled")
        OmegaConf.update(cfg, "optuna.n_trials", 0)
        OmegaConf.update(cfg, "dataset.demo_pool_size", 50)
        OmegaConf.update(cfg, "dataset.test_size", 10)
        
    elif mode == "pilot":
        OmegaConf.update(cfg, "mode.pilot", True)
        OmegaConf.update(cfg, "wandb.mode", "disabled")
        OmegaConf.update(cfg, "optuna.n_trials", 5)
        OmegaConf.update(cfg, "dataset.demo_pool_size", 100)
        OmegaConf.update(cfg, "dataset.test_size", 50)
        
    elif mode == "main":
        OmegaConf.update(cfg, "mode.main", True)
        OmegaConf.update(cfg, "wandb.mode", "online")
        
    # Get run_id and results_dir from config
    run_id = cfg.run.run_id
    results_dir = cfg.results_dir
    
    print(f"Run ID: {run_id}")
    print(f"Results dir: {results_dir}")
    
    # Build Hydra overrides for train.py
    hydra_overrides = [
        f"run.run_id={run_id}",
        f"results_dir={results_dir}",
        f"mode.sanity_check={cfg.mode.sanity_check}",
        f"mode.pilot={cfg.mode.pilot}",
        f"mode.main={cfg.mode.main}",
        f"wandb.mode={cfg.wandb.mode}",
        f"optuna.n_trials={cfg.optuna.n_trials}",
        f"dataset.demo_pool_size={cfg.dataset.demo_pool_size}",
        f"dataset.test_size={cfg.dataset.test_size}",
    ]
    
    # Build command to invoke train.py
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train"
    ] + hydra_overrides
    
    print(f"\nInvoking train.py...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run train.py as subprocess
    result = subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=Path(__file__).parent.parent
    )
    
    # Exit with same code as train.py
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
