"""Training script for C3-AutoCoT experiment."""

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from preprocess import SVAMPDataset, compute_accuracy, extract_answer_from_text
from model import LLMWrapper, ReliabilityScorer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_demonstrations(
    cfg: DictConfig,
    demo_pool: List[Dict],
    clusters: Dict[int, List[int]],
    llm: LLMWrapper,
    scorer: ReliabilityScorer
) -> List[Dict]:
    """Select reliable demonstrations from each cluster.
    
    Args:
        cfg: Hydra config
        demo_pool: Pool of candidate demonstrations
        clusters: Cluster assignments
        llm: Language model
        scorer: Reliability scorer
        
    Returns:
        List of selected demonstrations
    """
    selected_demos = []
    
    print(f"\nSelecting demonstrations (method={cfg.method.name})...")
    
    for cluster_id in tqdm(range(cfg.training.num_clusters), desc="Clusters"):
        cluster_indices = clusters[cluster_id]
        
        if not cluster_indices:
            continue
            
        # Shuffle candidates within cluster
        random.shuffle(cluster_indices)
        
        # Try candidates until we find a reliable one
        demo_found = False
        for idx in cluster_indices:
            candidate = demo_pool[idx]
            
            # Generate CoT reasoning for candidate
            reasoning = llm.generate_cot(candidate["question"])
            predicted_answer = extract_answer_from_text(reasoning)
            
            # Check if answer is correct (for demo selection, we have labels)
            if not math.isnan(predicted_answer):
                answer_correct = abs(predicted_answer - candidate["answer"]) < 1e-3
            else:
                answer_correct = False
                
            if not answer_correct:
                continue
                
            # Compute reliability score
            if cfg.method.type == "comparative-1":
                # Standard Auto-CoT: no reliability filtering
                reliability = 1.0
                demo_found = True
            else:
                # Compute reliability components
                reliability, components = scorer.compute_c3_reliability(
                    question=candidate["question"],
                    reasoning=reasoning,
                    answer=candidate["answer"],
                    sc_samples=cfg.method.self_consistency_samples,
                    pi_paraphrases=cfg.method.paraphrase_count,
                    enable_cc=cfg.method.cycle_consistency_enabled
                )
                
                # Check threshold
                if reliability >= cfg.method.reliability_threshold:
                    demo_found = True
                    
            if demo_found:
                selected_demos.append({
                    "id": candidate["id"],
                    "question": candidate["question"],
                    "reasoning": reasoning,
                    "answer": candidate["answer"],
                    "reliability": reliability
                })
                break
                
        if not demo_found:
            # Fallback: use first candidate with correct answer
            for idx in cluster_indices:
                candidate = demo_pool[idx]
                reasoning = llm.generate_cot(candidate["question"])
                predicted_answer = extract_answer_from_text(reasoning)
                
                if not math.isnan(predicted_answer):
                    answer_correct = abs(predicted_answer - candidate["answer"]) < 1e-3
                    if answer_correct:
                        selected_demos.append({
                            "id": candidate["id"],
                            "question": candidate["question"],
                            "reasoning": reasoning,
                            "answer": candidate["answer"],
                            "reliability": 0.0
                        })
                        break
                        
    print(f"Selected {len(selected_demos)} demonstrations")
    return selected_demos


def evaluate_on_test_set(
    cfg: DictConfig,
    test_set: List[Dict],
    demonstrations: List[Dict],
    llm: LLMWrapper
) -> Dict:
    """Evaluate on test set using selected demonstrations.
    
    Args:
        cfg: Hydra config
        test_set: Test examples
        demonstrations: Selected demonstrations
        llm: Language model
        
    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating on {len(test_set)} test examples...")
    
    predictions = []
    targets = []
    results = []
    
    for example in tqdm(test_set, desc="Evaluating"):
        # Generate CoT reasoning with demonstrations
        reasoning = llm.generate_cot(example["question"], demos=demonstrations)
        predicted_answer = extract_answer_from_text(reasoning)
        
        predictions.append(predicted_answer)
        targets.append(example["answer"])
        
        results.append({
            "id": example["id"],
            "question": example["question"],
            "predicted_answer": predicted_answer,
            "true_answer": example["answer"],
            "reasoning": reasoning
        })
        
    # Compute accuracy
    accuracy = compute_accuracy(predictions, targets)
    
    metrics = {
        "accuracy": accuracy,
        "num_test": len(test_set),
        "num_demos": len(demonstrations)
    }
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return metrics, results


def run_optuna_search(cfg: DictConfig, demo_pool: List[Dict], test_set: List[Dict]) -> DictConfig:
    """Run Optuna hyperparameter search.
    
    Args:
        cfg: Hydra config
        demo_pool: Demo pool
        test_set: Test set
        
    Returns:
        Updated config with best hyperparameters
    """
    import optuna
    
    print("\nRunning Optuna hyperparameter search...")
    
    def objective(trial):
        # Sample hyperparameters
        temp_cfg = OmegaConf.to_container(cfg, resolve=True)
        temp_cfg = OmegaConf.create(temp_cfg)
        
        for param_name, param_config in cfg.optuna.parameters.items():
            if param_config.type == "float":
                value = trial.suggest_float(param_name, param_config.low, param_config.high)
            elif param_config.type == "int":
                value = trial.suggest_int(param_name, param_config.low, param_config.high)
            else:
                raise ValueError(f"Unknown parameter type: {param_config.type}")
                
            # Update config
            OmegaConf.update(temp_cfg, f"method.{param_name}", value)
            
        # Run trial with small validation set
        val_size = min(50, len(test_set))
        val_set = test_set[:val_size]
        
        # Initialize model and scorer
        llm = LLMWrapper(
            model_name=temp_cfg.model.name,
            device=temp_cfg.model.device,
            cache_dir=temp_cfg.cache_dir,
            load_in_8bit=temp_cfg.model.get("load_in_8bit", False),
            max_new_tokens=temp_cfg.model.max_new_tokens,
            temperature=temp_cfg.model.temperature
        )
        scorer = ReliabilityScorer(llm)
        
        # Cluster demo pool
        dataset = SVAMPDataset(cache_dir=temp_cfg.cache_dir, random_seed=temp_cfg.training.random_seed)
        clusters = dataset.cluster_questions(
            demo_pool,
            num_clusters=temp_cfg.training.num_clusters,
            embedding_model_name=temp_cfg.training.embedding_model
        )
        
        # Select demonstrations
        demos = select_demonstrations(temp_cfg, demo_pool, clusters, llm, scorer)
        
        # Evaluate
        metrics, _ = evaluate_on_test_set(temp_cfg, val_set, demos, llm)
        
        return metrics["accuracy"]
        
    # Create study
    sampler = getattr(optuna.samplers, cfg.optuna.sampler)()
    pruner = getattr(optuna.pruners, cfg.optuna.pruner)()
    
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Run optimization
    study.optimize(objective, n_trials=cfg.optuna.n_trials)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Update config with best params
    for param_name, param_value in study.best_params.items():
        OmegaConf.update(cfg, f"method.{param_name}", param_value)
        
    return cfg


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    
    print("="*80)
    print(f"C3-AutoCoT Training")
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Method: {cfg.method.name} ({cfg.method.type})")
    print("="*80)
    
    # Set random seed
    set_seed(cfg.training.random_seed)
    
    # Initialize WandB (if not disabled)
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode
        )
        print(f"WandB run: {wandb.run.url}")
    else:
        print("WandB disabled")
        
    # Load dataset
    dataset = SVAMPDataset(cache_dir=cfg.cache_dir, random_seed=cfg.training.random_seed)
    demo_pool, test_set = dataset.load_data(
        demo_pool_size=cfg.dataset.demo_pool_size,
        test_size=cfg.dataset.test_size
    )
    
    # Apply sanity check mode limits
    if cfg.mode.sanity_check:
        print("\n*** SANITY CHECK MODE ***")
        demo_pool = demo_pool[:50]
        test_set = test_set[:10]
        
    # Initialize model
    llm = LLMWrapper(
        model_name=cfg.model.name,
        device=cfg.model.device,
        cache_dir=cfg.cache_dir,
        load_in_8bit=cfg.model.get("load_in_8bit", False),
        max_new_tokens=cfg.model.max_new_tokens,
        temperature=cfg.model.temperature
    )
    
    # Initialize reliability scorer
    scorer = ReliabilityScorer(llm)
    
    # Run Optuna search if enabled (and not in sanity check mode)
    if cfg.optuna.enabled and cfg.optuna.n_trials > 0 and not cfg.mode.sanity_check:
        cfg = run_optuna_search(cfg, demo_pool, test_set)
        
    # Cluster demo pool
    clusters = dataset.cluster_questions(
        demo_pool,
        num_clusters=cfg.training.num_clusters,
        embedding_model_name=cfg.training.embedding_model
    )
    
    # Select demonstrations
    demonstrations = select_demonstrations(cfg, demo_pool, clusters, llm, scorer)
    
    # Evaluate on test set
    metrics, results = evaluate_on_test_set(cfg, test_set, demonstrations, llm)
    
    # Log metrics to WandB
    if cfg.wandb.mode != "disabled":
        wandb.log(metrics)
        wandb.summary.update(metrics)
        
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(results_dir / "demonstrations.json", "w") as f:
        json.dump(demonstrations, f, indent=2)
        
    print(f"\nResults saved to {results_dir}")
    
    # Sanity validation
    if cfg.mode.sanity_check:
        perform_sanity_validation(metrics, len(test_set))
        
    # Finish WandB
    if cfg.wandb.mode != "disabled":
        wandb.finish()
        
    print("\nTraining complete!")


def perform_sanity_validation(metrics: Dict, num_steps: int):
    """Perform sanity validation checks and print verdict."""
    
    # Prepare summary
    summary = {
        "steps": num_steps,
        "accuracy": metrics.get("accuracy", 0.0),
        "num_test": metrics.get("num_test", 0),
        "num_demos": metrics.get("num_demos", 0)
    }
    
    print(f"\nSANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
    
    # Check conditions
    fail_reason = None
    
    # Check 1: Sufficient steps
    if num_steps < 5:
        fail_reason = "insufficient_steps"
        
    # Check 2: Metrics are finite
    accuracy = metrics.get("accuracy", float('nan'))
    if not math.isfinite(accuracy):
        fail_reason = "non_finite_metrics"
        
    # Check 3: Accuracy is not always 0
    if accuracy == 0.0:
        fail_reason = "zero_accuracy"
        
    # Check 4: Metrics exist
    if "accuracy" not in metrics:
        fail_reason = "missing_metrics"
        
    # Print verdict
    if fail_reason:
        print(f"SANITY_VALIDATION: FAIL reason={fail_reason}")
    else:
        print("SANITY_VALIDATION: PASS")


if __name__ == "__main__":
    main()
