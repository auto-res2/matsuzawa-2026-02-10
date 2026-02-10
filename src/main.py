"""
Main Orchestration Script
Handles mode selection and invokes training script.
"""
import sys
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for experiment execution.
    Applies mode-specific overrides and invokes training.
    """
    
    # Parse command-line flags
    sanity_check = '--sanity_check' in sys.argv
    is_main = '--main' in sys.argv
    is_pilot = '--pilot' in sys.argv
    
    # Apply mode-specific overrides
    if sanity_check:
        print("Running in SANITY_CHECK mode")
        cfg.wandb.mode = "disabled"
        cfg.sanity_check = True
        if 'optuna' in cfg and cfg.optuna is not None:
            cfg.optuna.n_trials = 0
    elif is_pilot:
        print("Running in PILOT mode")
        # Pilot mode: reduced but not minimal
        cfg.wandb.mode = "disabled"
        cfg.sanity_check = False
        if 'optuna' in cfg and cfg.optuna is not None:
            cfg.optuna.n_trials = min(cfg.optuna.get('n_trials', 10), 5)
    elif is_main:
        print("Running in MAIN mode")
        cfg.wandb.mode = "online"
        cfg.sanity_check = False
    else:
        print("No mode specified, defaulting to SANITY_CHECK")
        cfg.wandb.mode = "disabled"
        cfg.sanity_check = True
    
    print(f"Configuration:")
    print(f"  Run ID: {cfg.run.run_id}")
    print(f"  Results Dir: {cfg.results_dir}")
    print(f"  WandB Mode: {cfg.wandb.mode}")
    print(f"  Sanity Check: {cfg.get('sanity_check', False)}")
    
    # Import and run training
    # Since both functions are hydra-decorated, we need to invoke the inner function directly
    import src.train
    
    # Call the train main function directly with the config
    src.train.main(cfg)


if __name__ == "__main__":
    main()
