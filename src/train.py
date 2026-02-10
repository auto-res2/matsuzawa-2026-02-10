"""
Training script for Auto-CoT experiments.
Executed as a single run by main.py.
"""
import os
import json
import sys
from typing import List, Dict, Any
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch

from src.preprocess import (
    load_svamp_dataset,
    cluster_questions,
    format_cot_prompt,
    extract_answer_from_text,
    compare_answers,
    save_demo_pool
)
from src.model import create_model


class MetricsTracker:
    """Track metrics for sanity validation."""
    
    def __init__(self):
        self.steps = []
        self.losses = []
        self.accuracies = []
        self.all_metrics = []
    
    def log(self, step: int, metrics: Dict[str, Any]):
        """Log a step's metrics."""
        self.steps.append(step)
        
        if 'loss' in metrics:
            self.losses.append(metrics['loss'])
        if 'accuracy' in metrics:
            self.accuracies.append(metrics['accuracy'])
        
        self.all_metrics.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary for validation."""
        summary = {
            "steps": len(self.steps),
            "loss_start": self.losses[0] if self.losses else None,
            "loss_end": self.losses[-1] if self.losses else None,
            "accuracy_min": min(self.accuracies) if self.accuracies else None,
            "accuracy_max": max(self.accuracies) if self.accuracies else None,
        }
        return summary
    
    def validate(self, is_sanity_check: bool = False) -> tuple[bool, str]:
        """
        Validate metrics for sanity check.
        
        Returns:
            (passed, reason)
        """
        if not is_sanity_check:
            return True, "not_sanity_check"
        
        # Check minimum steps
        if len(self.steps) < 5:
            return False, f"insufficient_steps_{len(self.steps)}"
        
        # Check for missing metrics
        if not self.losses and not self.accuracies:
            return False, "missing_metrics"
        
        # Check for NaN/inf in losses
        if self.losses:
            for loss in self.losses:
                if not isinstance(loss, (int, float)) or loss != loss or abs(loss) == float('inf'):
                    return False, "nan_or_inf_loss"
            
            # Check loss decrease
            if self.losses[-1] > self.losses[0]:
                return False, f"loss_increased_{self.losses[0]:.4f}_to_{self.losses[-1]:.4f}"
        
        # Check accuracy is not always 0
        if self.accuracies:
            if max(self.accuracies) == 0:
                return False, "accuracy_always_zero"
        
        return True, "pass"


def train_single_run(cfg: DictConfig) -> Dict[str, Any]:
    """
    Execute a single training run.
    
    Returns:
        Final metrics dictionary
    """
    print("=" * 80)
    print(f"Starting run: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method.type}")
    print("=" * 80)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Determine if this is sanity check mode
    is_sanity_check = cfg.run.wandb.mode == "disabled"
    
    # Initialize WandB if not disabled
    if cfg.run.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.run.wandb.entity,
            project=cfg.run.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.run.wandb.mode
        )
        print(f"WandB run initialized: {wandb.run.url}")
    else:
        print("WandB disabled (sanity_check mode)")
    
    # Load dataset
    print("\nLoading SVAMP dataset...")
    train_data, test_data = load_svamp_dataset(
        cache_dir=cfg.run.dataset.cache_dir,
        seed=cfg.run.dataset.seed
    )
    
    # Use first demo_pool_size examples for demo pool
    demo_pool = train_data[:cfg.run.dataset.demo_pool_size]
    
    # Use test_size examples for evaluation
    test_data = test_data[:cfg.run.training.num_test_questions]
    
    print(f"Demo pool size: {len(demo_pool)}")
    print(f"Test set size: {len(test_data)}")
    
    # Cluster demo pool
    print("\nClustering demo pool questions...")
    demo_questions = [item['question'] for item in demo_pool]
    cluster_labels, embeddings = cluster_questions(
        demo_questions,
        num_clusters=cfg.run.method.num_clusters,
        cache_dir=cfg.run.dataset.cache_dir
    )
    
    print(f"Clustered into {cfg.run.method.num_clusters} clusters")
    
    # Create model
    print("\nLoading model...")
    model_config = {
        'model_name': cfg.run.model.name,
        'cache_dir': cfg.run.model.cache_dir,
        'device': cfg.run.model.device,
        'max_new_tokens': cfg.run.model.max_new_tokens,
        'load_in_8bit': cfg.run.model.get('load_in_8bit', False)
    }
    
    model = create_model(cfg.run.method.type, model_config)
    
    # Select demonstrations
    print("\nSelecting demonstrations...")
    select_demos_kwargs = {
        'demo_pool': demo_pool,
        'cluster_labels': cluster_labels,
        'num_clusters': cfg.run.method.num_clusters,
        'reliability_threshold': cfg.run.method.reliability_threshold,
        'self_consistency_config': OmegaConf.to_container(cfg.run.method.self_consistency),
        'paraphrase_invariance_config': OmegaConf.to_container(cfg.run.method.paraphrase_invariance),
        'cycle_consistency_config': OmegaConf.to_container(cfg.run.method.cycle_consistency)
    }
    
    # Add max_candidates_per_cluster if it exists in config
    if 'max_candidates_per_cluster' in cfg.run.method:
        select_demos_kwargs['max_candidates_per_cluster'] = cfg.run.method.max_candidates_per_cluster
    
    selected_demos = model.select_demonstrations(**select_demos_kwargs)
    
    print(f"\nSelected {len(selected_demos)} demonstrations")
    
    # Save demo pool
    results_dir = cfg.results_dir
    os.makedirs(f"{results_dir}/{cfg.run.run_id}", exist_ok=True)
    save_demo_pool(selected_demos, f"{results_dir}/{cfg.run.run_id}/demos.json")
    
    if len(selected_demos) == 0:
        print("WARNING: No demonstrations selected! Using empty demo set.")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80)
    
    correct = 0
    total = 0
    
    batch_size = cfg.run.training.batch_size
    num_test = len(test_data)
    
    for batch_start in range(0, num_test, batch_size):
        batch_end = min(batch_start + batch_size, num_test)
        batch = test_data[batch_start:batch_end]
        
        batch_correct = 0
        
        for test_item in batch:
            question = test_item['question']
            true_answer = test_item['answer']
            
            # Format prompt with demonstrations
            prompt = format_cot_prompt(question, selected_demos[:cfg.run.training.num_demos])
            
            # Generate answer (greedy decoding)
            outputs = model.generate(prompt, temperature=0.0)
            predicted_answer = extract_answer_from_text(outputs[0])
            
            # Check correctness
            is_correct = compare_answers(predicted_answer, true_answer)
            if is_correct:
                correct += 1
                batch_correct += 1
            
            total += 1
            
            if total <= 3 or (is_sanity_check and total <= 10):
                print(f"\nQ: {question[:80]}...")
                print(f"True: {true_answer}, Pred: {predicted_answer}, Correct: {is_correct}")
        
        # Log metrics for this batch
        batch_num = batch_start // batch_size + 1
        current_accuracy = correct / total
        
        # Compute a pseudo-loss (inverse of accuracy, shifted)
        pseudo_loss = 1.0 - current_accuracy
        
        step_metrics = {
            'accuracy': current_accuracy,
            'loss': pseudo_loss,
            'batch': batch_num,
            'correct': correct,
            'total': total
        }
        
        metrics_tracker.log(total, step_metrics)
        
        if cfg.run.wandb.mode != "disabled":
            wandb.log(step_metrics, step=total)
        
        print(f"\nBatch {batch_num}/{(num_test + batch_size - 1) // batch_size}: Accuracy = {current_accuracy:.4f} ({correct}/{total})")
    
    # Final metrics
    final_accuracy = correct / total if total > 0 else 0.0
    final_metrics = {
        'final_accuracy': final_accuracy,
        'correct': correct,
        'total': total,
        'num_demos_selected': len(selected_demos),
        'avg_demo_reliability': sum(d.get('reliability', 0) for d in selected_demos) / len(selected_demos) if selected_demos else 0
    }
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Accuracy: {final_accuracy:.4f} ({correct}/{total})")
    print(f"Demos selected: {len(selected_demos)}")
    print("=" * 80)
    
    # Save to WandB summary
    if cfg.run.wandb.mode != "disabled":
        for key, value in final_metrics.items():
            wandb.summary[key] = value
        wandb.finish()
    
    # Save metrics locally
    with open(f"{results_dir}/{cfg.run.run_id}/metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Sanity validation
    if is_sanity_check:
        passed, reason = metrics_tracker.validate(is_sanity_check=True)
        summary = metrics_tracker.get_summary()
        
        print(f"\nSANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
        
        if passed:
            print("SANITY_VALIDATION: PASS")
        else:
            print(f"SANITY_VALIDATION: FAIL reason={reason}")
    
    return final_metrics


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    # Set run config from the 'run' parameter
    if 'run' in cfg and cfg.run:
        run_config_path = f"../config/runs/{cfg.run}.yaml"
        # The run config is already loaded by Hydra
        pass
    
    # Execute training
    metrics = train_single_run(cfg)
    
    return metrics


if __name__ == "__main__":
    main()
