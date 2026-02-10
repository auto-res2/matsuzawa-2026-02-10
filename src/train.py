"""
Training Script: Demo Construction and Evaluation
Executes a single run with the specified method configuration.
"""
import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from typing import List, Dict, Tuple
import numpy as np

from src.model import create_model
from src.preprocess import (
    load_svamp_dataset,
    prepare_demo_pool,
    extract_numeric_answer,
    compute_self_consistency_score,
    compute_paraphrase_invariance_score,
    compute_cycle_consistency_score,
    generate_paraphrases,
    save_demonstrations,
)


def select_demonstrations(
    clusters: Dict[int, List[Dict]],
    model,
    cfg: DictConfig,
    sanity_check: bool = False
) -> Tuple[List[Dict], int]:
    """
    Select demonstrations from each cluster based on reliability scores.
    """
    selected_demos = []
    total_steps = 0
    
    # Sanity check mode: limit processing
    max_clusters = 2 if sanity_check else len(clusters)
    max_candidates_per_cluster = 2 if sanity_check else cfg.training.max_demo_generation_samples
    
    for cluster_id in sorted(clusters.keys())[:max_clusters]:
        cluster_examples = clusters[cluster_id]
        
        print(f"\n=== Processing Cluster {cluster_id} ({len(cluster_examples)} examples) ===")
        
        best_demo = None
        best_score = -1
        
        # Try up to max_candidates_per_cluster examples from this cluster
        for candidate_idx, candidate in enumerate(cluster_examples[:max_candidates_per_cluster]):
            question = candidate['question']
            ground_truth_answer = candidate['answer']
            
            print(f"  Candidate {candidate_idx + 1}/{min(len(cluster_examples), max_candidates_per_cluster)}: {question[:60]}...")
            
            # Step 1: Generate self-consistency samples
            num_sc_samples = 2 if sanity_check else cfg.method.num_self_consistency_samples
            cot_outputs = model.generate(
                f"Question: {question}\nAnswer: Let's solve this step by step.\n",
                num_samples=num_sc_samples
            )
            total_steps += num_sc_samples
            
            r_sc = compute_self_consistency_score(cot_outputs)
            print(f"    r_sc = {r_sc:.3f}")
            
            # Step 2: Paraphrase invariance (if enabled)
            r_pi = 1.0
            if cfg.method.use_paraphrase_invariance:
                num_para = 1 if sanity_check else cfg.method.num_paraphrase_samples
                paraphrases = generate_paraphrases(question, num_para)
                paraphrase_outputs = []
                
                for para in paraphrases:
                    para_outs = model.generate(
                        f"Question: {para}\nAnswer: Let's solve this step by step.\n",
                        num_samples=2 if sanity_check else 3
                    )
                    paraphrase_outputs.append(para_outs)
                    total_steps += len(para_outs)
                
                r_pi = compute_paraphrase_invariance_score(cot_outputs, paraphrase_outputs)
                print(f"    r_pi = {r_pi:.3f}")
            
            # Step 3: Cycle consistency (if enabled)
            r_cc = 1.0
            if cfg.method.use_cycle_consistency:
                # Use the most common rationale
                best_rationale = cot_outputs[0] if cot_outputs else ""
                predicted_answer = extract_numeric_answer(best_rationale)
                
                if predicted_answer is not None:
                    reconstructed_q = model.generate_question_from_rationale(
                        best_rationale,
                        str(predicted_answer)
                    )
                    r_cc = compute_cycle_consistency_score(question, best_rationale, reconstructed_q)
                    total_steps += 1
                    print(f"    r_cc = {r_cc:.3f}")
            
            # Compute total reliability score
            reliability = r_sc * r_pi * r_cc
            print(f"    Total reliability = {reliability:.3f}")
            
            # Check if this candidate passes the threshold
            if reliability >= cfg.method.reliability_threshold:
                if reliability > best_score:
                    best_score = reliability
                    best_demo = {
                        'question': question,
                        'rationale': cot_outputs[0] if cot_outputs else "",
                        'answer': ground_truth_answer,
                        'reliability_score': reliability,
                        'r_sc': r_sc,
                        'r_pi': r_pi,
                        'r_cc': r_cc,
                    }
                    print(f"    ✓ New best demo for cluster {cluster_id}")
        
        if best_demo:
            selected_demos.append(best_demo)
            print(f"  Selected demo with score {best_score:.3f}")
        else:
            print(f"  No demo passed threshold in cluster {cluster_id}")
    
    print(f"\n=== Demo Selection Complete: {len(selected_demos)}/{max_clusters} clusters ===")
    print(f"Total training steps: {total_steps}")
    
    return selected_demos, total_steps


def evaluate_on_test_set(
    test_data: List[Dict],
    demonstrations: List[Dict],
    model,
    cfg: DictConfig,
    sanity_check: bool = False
) -> Tuple[float, int]:
    """
    Evaluate the selected demonstrations on test set.
    Returns (accuracy, num_eval_steps).
    """
    # Limit test set in sanity check mode
    if sanity_check:
        test_data = test_data[:5]  # Just 5 examples for sanity check
    else:
        test_data = test_data[:cfg.dataset.test_size]
    
    correct = 0
    total = 0
    eval_steps = 0
    
    print(f"\n=== Evaluating on {len(test_data)} test examples ===")
    
    for idx, test_example in enumerate(test_data):
        question = test_example['question']
        ground_truth = test_example['answer']
        
        # Generate answer with demonstrations (greedy decoding)
        outputs = model.generate_cot_rationale(
            question,
            demonstrations,
            num_samples=1
        )
        eval_steps += 1
        
        if outputs:
            predicted_answer = extract_numeric_answer(outputs[0])
            if predicted_answer is not None:
                is_correct = abs(predicted_answer - ground_truth) < 1e-6
                if is_correct:
                    correct += 1
                total += 1
                
                if idx < 3 or sanity_check:  # Show first few examples
                    print(f"  [{idx+1}] GT: {ground_truth}, Pred: {predicted_answer} {'✓' if is_correct else '✗'}")
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n=== Final Accuracy: {accuracy:.4f} ({correct}/{total}) ===")
    print(f"Evaluation steps: {eval_steps}")
    
    return accuracy, eval_steps


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    
    print("=" * 80)
    print(f"Starting run: {cfg.run.run_id}")
    print(f"Method: {cfg.method.name} ({cfg.method.type})")
    print("=" * 80)
    
    # Determine mode
    sanity_check = cfg.get('sanity_check', False)
    mode_str = "SANITY_CHECK" if sanity_check else "FULL"
    print(f"Mode: {mode_str}")
    
    # Initialize WandB if not disabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow"
        )
        print(f"WandB initialized: {wandb.run.get_url()}")
    else:
        print("WandB disabled")
    
    # Load dataset
    print("\nLoading SVAMP dataset...")
    train_data, test_data = load_svamp_dataset(cfg.dataset.cache_dir)
    print(f"Loaded {len(train_data)} train, {len(test_data)} test examples")
    
    # Prepare demo pool and clustering
    print(f"\nPreparing demo pool (first {cfg.dataset.demo_pool_size} examples)...")
    clusters = prepare_demo_pool(
        train_data,
        cfg.dataset.demo_pool_size,
        cfg.dataset.num_clusters,
        cfg.dataset.embedding_model,
        cfg.model.cache_dir
    )
    print(f"Created {len(clusters)} clusters")
    
    # Load model
    print("\nLoading model...")
    model = create_model(cfg)
    
    # Select demonstrations
    print("\n" + "=" * 80)
    print("PHASE 1: Demo Construction")
    print("=" * 80)
    demonstrations, train_steps = select_demonstrations(clusters, model, cfg, sanity_check)
    
    # Save demonstrations
    demo_path = os.path.join(cfg.results_dir, cfg.run.run_id, "demonstrations.json")
    save_demonstrations(demonstrations, demo_path)
    print(f"\nSaved {len(demonstrations)} demonstrations to {demo_path}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("PHASE 2: Test Set Evaluation")
    print("=" * 80)
    accuracy, eval_steps = evaluate_on_test_set(test_data, demonstrations, model, cfg, sanity_check)
    
    total_steps = train_steps + eval_steps
    
    # Log metrics to WandB
    metrics = {
        'accuracy': accuracy,
        'num_demonstrations': len(demonstrations),
        'total_steps': total_steps,
        'train_steps': train_steps,
        'eval_steps': eval_steps,
    }
    
    if cfg.wandb.mode != "disabled":
        wandb.log(metrics)
        for key, value in metrics.items():
            wandb.summary[key] = value
        print(f"\nLogged metrics to WandB")
    
    # Save metrics locally
    metrics_path = os.path.join(cfg.results_dir, cfg.run.run_id, "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Sanity validation
    if sanity_check:
        print("\n" + "=" * 80)
        print("SANITY VALIDATION")
        print("=" * 80)
        
        # Check conditions
        passes = []
        reasons = []
        
        # Condition 1: At least 5 steps
        if total_steps >= 5:
            passes.append(True)
            print(f"✓ Total steps: {total_steps} >= 5")
        else:
            passes.append(False)
            reasons.append(f"insufficient_steps_{total_steps}")
            print(f"✗ Total steps: {total_steps} < 5")
        
        # Condition 2: Metrics are finite
        if np.isfinite(accuracy):
            passes.append(True)
            print(f"✓ Accuracy is finite: {accuracy}")
        else:
            passes.append(False)
            reasons.append("non_finite_metrics")
            print(f"✗ Accuracy is not finite: {accuracy}")
        
        # Condition 3: Accuracy is not always 0
        if accuracy > 0:
            passes.append(True)
            print(f"✓ Accuracy > 0: {accuracy:.4f}")
        else:
            passes.append(False)
            reasons.append("zero_accuracy")
            print(f"✗ Accuracy is 0")
        
        # Summary
        validation_summary = {
            "steps": total_steps,
            "accuracy": accuracy,
            "num_demos": len(demonstrations),
        }
        print(f"\nSANITY_VALIDATION_SUMMARY: {json.dumps(validation_summary)}")
        
        # Verdict
        if all(passes):
            print("SANITY_VALIDATION: PASS")
        else:
            reason = reasons[0] if reasons else "unknown"
            print(f"SANITY_VALIDATION: FAIL reason={reason}")
    
    # Finish WandB
    if cfg.wandb.mode != "disabled":
        wandb.finish()
    
    print("\n" + "=" * 80)
    print(f"Run {cfg.run.run_id} completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
