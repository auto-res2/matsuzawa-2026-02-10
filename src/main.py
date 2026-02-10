"""
Main orchestrator for running experiments.
Handles mode overrides and delegates to train.py.
"""
import sys
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import subprocess


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main orchestrator that applies mode overrides and runs training.
    """
    # Determine mode from command line arguments
    mode = None
    if '--sanity_check' in sys.argv:
        mode = 'sanity_check'
    elif '--pilot' in sys.argv:
        mode = 'pilot'
    elif '--main' in sys.argv:
        mode = 'main'
    
    print(f"Running in mode: {mode}")
    
    # Apply mode-specific overrides
    if mode and mode in cfg.mode:
        mode_overrides = cfg.mode[mode]
        print(f"Applying {mode} mode overrides:")
        print(OmegaConf.to_yaml(mode_overrides))
        
        # Merge mode overrides into run config
        cfg.run = OmegaConf.merge(cfg.run, mode_overrides)
    
    # Since we're already in the Hydra context, we can directly import and call train
    from src.train import train_single_run
    
    # Execute training
    metrics = train_single_run(cfg)
    
    return metrics


if __name__ == "__main__":
    main()
