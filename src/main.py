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


def parse_custom_flags():
    """Parse custom flags and convert to Hydra overrides."""
    # Check for custom flags and remove them from sys.argv
    mode = None
    flags_to_remove = []
    
    for i, arg in enumerate(sys.argv):
        if arg == '--sanity_check':
            mode = 'sanity_check'
            flags_to_remove.append(i)
        elif arg == '--pilot':
            mode = 'pilot'
            flags_to_remove.append(i)
        elif arg == '--main':
            mode = 'main'
            flags_to_remove.append(i)
    
    # Remove flags in reverse order to maintain indices
    for i in reversed(flags_to_remove):
        sys.argv.pop(i)
    
    # Add Hydra override if mode was detected
    if mode:
        sys.argv.append(f'+exec_mode={mode}')
    
    return mode


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main orchestrator that applies mode overrides and runs training.
    """
    # Load run configuration if 'run' parameter is provided
    if 'run' in cfg and cfg.run != '???':
        # The run config should be loaded by Hydra via defaults
        # But if it's still a string, we need to load it manually
        if isinstance(cfg.run, str):
            run_id = cfg.run
            run_config_path = os.path.join(
                Path(__file__).parent.parent,
                "config",
                "runs",
                f"{run_id}.yaml"
            )
            
            if os.path.exists(run_config_path):
                run_config = OmegaConf.load(run_config_path)
                cfg = OmegaConf.merge(cfg, {"run": run_config})
            else:
                raise ValueError(f"Run configuration not found: {run_config_path}")
    
    # Get mode from config (set via custom flag conversion)
    mode = cfg.get('exec_mode', None)
    
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
    # Parse custom flags before Hydra initialization
    parse_custom_flags()
    main()
