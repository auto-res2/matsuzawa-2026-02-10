"""
Evaluation script for aggregating results and creating comparison figures.
Independent script that fetches WandB runs and generates visualizations.
"""
import os
import json
import argparse
from typing import List, Dict, Any
import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def fetch_wandb_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.
    
    Returns:
        Dictionary with 'config', 'summary', and 'history'
    """
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get config
        config = dict(run.config)
        
        # Get summary metrics
        summary = dict(run.summary)
        
        # Get history (logged metrics over time)
        history = run.history()
        history_data = history.to_dict('records') if not history.empty else []
        
        return {
            'config': config,
            'summary': summary,
            'history': history_data,
            'url': run.url
        }
    except Exception as e:
        print(f"Warning: Could not fetch WandB data for {run_id}: {e}")
        return {
            'config': {},
            'summary': {},
            'history': [],
            'url': None
        }


def load_local_metrics(results_dir: str, run_id: str) -> Dict[str, Any]:
    """
    Load metrics from local JSON file as fallback.
    """
    metrics_path = f"{results_dir}/{run_id}/metrics.json"
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    return {}


def create_accuracy_comparison_plot(
    results: Dict[str, Dict],
    output_path: str
):
    """
    Create a bar chart comparing final accuracy across runs.
    """
    run_ids = []
    accuracies = []
    
    for run_id, data in results.items():
        run_ids.append(run_id)
        
        # Try to get accuracy from summary or local metrics
        accuracy = data['summary'].get('final_accuracy', 
                                       data['local_metrics'].get('final_accuracy', 0))
        accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(run_ids, accuracies, color=['#2ecc71' if 'proposed' in rid else '#3498db' for rid in run_ids])
    
    plt.xlabel('Run ID', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {output_path}")


def create_training_curves_plot(
    results: Dict[str, Dict],
    output_path: str
):
    """
    Create training curves showing accuracy over time.
    """
    plt.figure(figsize=(12, 6))
    
    for run_id, data in results.items():
        history = data.get('history', [])
        
        if history:
            steps = [h.get('total', h.get('_step', i)) for i, h in enumerate(history)]
            accuracies = [h.get('accuracy', 0) for h in history]
            
            label = run_id
            color = '#2ecc71' if 'proposed' in run_id else '#3498db'
            marker = 'o' if 'proposed' in run_id else 's'
            
            plt.plot(steps, accuracies, marker=marker, label=label, 
                    color=color, linewidth=2, markersize=6)
    
    plt.xlabel('Test Examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy over Test Set', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {output_path}")


def create_per_run_plots(
    run_id: str,
    data: Dict,
    output_dir: str
):
    """
    Create individual plots for a single run.
    """
    history = data.get('history', [])
    
    if not history:
        print(f"No history data for {run_id}, skipping per-run plots")
        return
    
    # Accuracy over time
    plt.figure(figsize=(10, 6))
    steps = [h.get('total', h.get('_step', i)) for i, h in enumerate(history)]
    accuracies = [h.get('accuracy', 0) for h in history]
    
    plt.plot(steps, accuracies, marker='o', linewidth=2, markersize=6, color='#2ecc71')
    plt.xlabel('Test Examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{run_id}: Accuracy Progress', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    output_path = f"{output_dir}/{run_id}_accuracy.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {output_path}")


def evaluate_runs(
    results_dir: str,
    run_ids: List[str],
    wandb_entity: str,
    wandb_project: str
) -> Dict[str, Any]:
    """
    Evaluate multiple runs and generate comparison metrics and figures.
    """
    print("=" * 80)
    print("Evaluation Script")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Run IDs: {run_ids}")
    print(f"WandB: {wandb_entity}/{wandb_project}")
    print("=" * 80)
    
    # Fetch data for each run
    results = {}
    
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        
        # Try WandB first
        wandb_data = fetch_wandb_run_data(wandb_entity, wandb_project, run_id)
        
        # Load local metrics as fallback
        local_metrics = load_local_metrics(results_dir, run_id)
        
        results[run_id] = {
            'config': wandb_data['config'],
            'summary': wandb_data['summary'],
            'history': wandb_data['history'],
            'local_metrics': local_metrics,
            'url': wandb_data['url']
        }
        
        # Export per-run metrics
        run_output_dir = f"{results_dir}/{run_id}"
        os.makedirs(run_output_dir, exist_ok=True)
        
        per_run_metrics = {
            'run_id': run_id,
            'final_accuracy': results[run_id]['summary'].get('final_accuracy',
                                                             local_metrics.get('final_accuracy', 0)),
            'correct': results[run_id]['summary'].get('correct',
                                                       local_metrics.get('correct', 0)),
            'total': results[run_id]['summary'].get('total',
                                                     local_metrics.get('total', 0)),
            'num_demos_selected': results[run_id]['summary'].get('num_demos_selected',
                                                                  local_metrics.get('num_demos_selected', 0)),
            'wandb_url': results[run_id]['url']
        }
        
        metrics_path = f"{run_output_dir}/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(per_run_metrics, f, indent=2)
        
        print(f"Exported: {metrics_path}")
        
        # Create per-run figures
        create_per_run_plots(run_id, results[run_id], run_output_dir)
    
    # Create comparison directory
    comparison_dir = f"{results_dir}/comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract metrics for comparison
    metrics_by_run = {}
    
    for run_id, data in results.items():
        final_accuracy = data['summary'].get('final_accuracy',
                                             data['local_metrics'].get('final_accuracy', 0))
        metrics_by_run[run_id] = {
            'accuracy': final_accuracy,
            'correct': data['summary'].get('correct', data['local_metrics'].get('correct', 0)),
            'total': data['summary'].get('total', data['local_metrics'].get('total', 0))
        }
    
    # Identify best proposed and baseline
    proposed_runs = {k: v for k, v in metrics_by_run.items() if 'proposed' in k}
    baseline_runs = {k: v for k, v in metrics_by_run.items() if 'comparative' in k}
    
    best_proposed = max(proposed_runs.items(), key=lambda x: x[1]['accuracy']) if proposed_runs else (None, {'accuracy': 0})
    best_baseline = max(baseline_runs.items(), key=lambda x: x[1]['accuracy']) if baseline_runs else (None, {'accuracy': 0})
    
    gap = best_proposed[1]['accuracy'] - best_baseline[1]['accuracy']
    
    # Aggregated metrics
    aggregated_metrics = {
        'primary_metric': 'accuracy',
        'metrics': metrics_by_run,
        'best_proposed': {
            'run_id': best_proposed[0],
            'accuracy': best_proposed[1]['accuracy']
        },
        'best_baseline': {
            'run_id': best_baseline[0],
            'accuracy': best_baseline[1]['accuracy']
        },
        'gap': gap
    }
    
    # Save aggregated metrics
    agg_path = f"{comparison_dir}/aggregated_metrics.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    print(f"\nExported: {agg_path}")
    
    # Create comparison figures
    print("\nGenerating comparison figures...")
    
    create_accuracy_comparison_plot(
        results,
        f"{comparison_dir}/accuracy_comparison.png"
    )
    
    create_training_curves_plot(
        results,
        f"{comparison_dir}/training_curves.png"
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Best Proposed: {best_proposed[0]} - Accuracy: {best_proposed[1]['accuracy']:.4f}")
    print(f"Best Baseline: {best_baseline[0]} - Accuracy: {best_baseline[1]['accuracy']:.4f}")
    print(f"Gap: {gap:+.4f}")
    print("=" * 80)
    
    return aggregated_metrics


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate experiment runs')
    parser.add_argument('results_dir', type=str, help='Results directory')
    parser.add_argument('run_ids', type=str, help='JSON string list of run IDs')
    parser.add_argument('--wandb_entity', type=str, default='airas', help='WandB entity')
    parser.add_argument('--wandb_project', type=str, default='2026-02-10-test', help='WandB project')
    
    args = parser.parse_args()
    
    # Parse run_ids from JSON string
    run_ids = json.loads(args.run_ids)
    
    # Run evaluation
    evaluate_runs(
        results_dir=args.results_dir,
        run_ids=run_ids,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project
    )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
