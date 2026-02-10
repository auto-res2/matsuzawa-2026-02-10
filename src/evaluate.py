"""
Evaluation Script: Fetch WandB runs and generate comparison reports
"""
import os
import json
import argparse
from typing import List, Dict
import wandb
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate and compare runs')
    parser.add_argument('results_dir', type=str, help='Results directory path')
    parser.add_argument('run_ids', type=str, help='JSON string list of run IDs')
    parser.add_argument('--wandb_entity', type=str, default='airas', help='WandB entity')
    parser.add_argument('--wandb_project', type=str, default='2026-02-10-test', help='WandB project')
    return parser.parse_args()


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch run data from WandB API.
    Returns dict with config, summary, and history.
    """
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        data = {
            'run_id': run_id,
            'config': dict(run.config),
            'summary': dict(run.summary),
            'history': run.history().to_dict('records') if hasattr(run, 'history') else []
        }
        
        return data
    except Exception as e:
        print(f"Warning: Could not fetch run {run_id} from WandB: {e}")
        # Try to load from local files
        return load_local_run_data(run_id)


def load_local_run_data(run_id: str) -> Dict:
    """Load run data from local files if WandB is unavailable."""
    # This is a fallback for when WandB is disabled or unavailable
    return {
        'run_id': run_id,
        'config': {},
        'summary': {},
        'history': []
    }


def export_run_metrics(run_data: Dict, results_dir: str):
    """Export per-run metrics to JSON."""
    run_id = run_data['run_id']
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    metrics_path = os.path.join(run_dir, 'metrics.json')
    
    # If local metrics already exist, merge with WandB data
    local_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            local_metrics = json.load(f)
    
    # Merge summary metrics
    metrics = {**local_metrics, **run_data['summary']}
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics for {run_id} to {metrics_path}")
    return metrics


def create_run_figures(run_data: Dict, results_dir: str):
    """Create per-run visualization figures."""
    run_id = run_data['run_id']
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    history = run_data.get('history', [])
    
    if not history:
        print(f"No history data for {run_id}, skipping figures")
        return []
    
    figures = []
    
    # Figure 1: Metrics over time (if available)
    if any('accuracy' in record for record in history):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = [i for i, _ in enumerate(history)]
        accuracies = [record.get('accuracy', None) for record in history]
        
        ax.plot(steps, accuracies, marker='o', label='Accuracy')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy over Time - {run_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig_path = os.path.join(run_dir, 'accuracy_over_time.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        figures.append(fig_path)
        print(f"Created figure: {fig_path}")
    
    return figures


def create_comparison_figures(all_metrics: Dict[str, Dict], results_dir: str):
    """Create comparison figures across runs."""
    comparison_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    figures = []
    
    # Figure 1: Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    run_ids = list(all_metrics.keys())
    accuracies = [metrics.get('accuracy', 0) for metrics in all_metrics.values()]
    
    # Color proposed vs comparative differently
    colors = []
    for run_id in run_ids:
        if 'proposed' in run_id:
            colors.append('#2ecc71')  # Green for proposed
        else:
            colors.append('#3498db')  # Blue for comparative
    
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across Methods')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Proposed'),
        Patch(facecolor='#3498db', label='Comparative')
    ]
    ax.legend(handles=legend_elements)
    
    fig_path = os.path.join(comparison_dir, 'accuracy_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    figures.append(fig_path)
    print(f"Created comparison figure: {fig_path}")
    
    # Figure 2: Number of demonstrations comparison
    if all('num_demonstrations' in metrics for metrics in all_metrics.values()):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        num_demos = [metrics.get('num_demonstrations', 0) for metrics in all_metrics.values()]
        
        bars = ax.bar(range(len(run_ids)), num_demos, color=colors)
        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels(run_ids, rotation=45, ha='right')
        ax.set_ylabel('Number of Demonstrations')
        ax.set_title('Number of Selected Demonstrations by Method')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, num_demos):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9)
        
        ax.legend(handles=legend_elements)
        
        fig_path = os.path.join(comparison_dir, 'num_demonstrations_comparison.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        figures.append(fig_path)
        print(f"Created comparison figure: {fig_path}")
    
    return figures


def compute_aggregated_metrics(all_metrics: Dict[str, Dict]) -> Dict:
    """Compute aggregated comparison metrics."""
    
    # Identify proposed vs baseline runs
    proposed_runs = {k: v for k, v in all_metrics.items() if 'proposed' in k}
    baseline_runs = {k: v for k, v in all_metrics.items() if 'comparative' in k or 'baseline' in k}
    
    # Extract accuracies
    proposed_accuracies = [m.get('accuracy', 0) for m in proposed_runs.values()]
    baseline_accuracies = [m.get('accuracy', 0) for m in baseline_runs.values()]
    
    best_proposed = max(proposed_accuracies) if proposed_accuracies else 0
    best_baseline = max(baseline_accuracies) if baseline_accuracies else 0
    
    gap = best_proposed - best_baseline
    relative_improvement = (gap / best_baseline * 100) if best_baseline > 0 else 0
    
    aggregated = {
        'primary_metric': 'accuracy',
        'metrics': all_metrics,
        'best_proposed': {
            'run_id': [k for k, v in proposed_runs.items() if v.get('accuracy', 0) == best_proposed][0] if proposed_runs else None,
            'accuracy': best_proposed
        },
        'best_baseline': {
            'run_id': [k for k, v in baseline_runs.items() if v.get('accuracy', 0) == best_baseline][0] if baseline_runs else None,
            'accuracy': best_baseline
        },
        'gap': gap,
        'relative_improvement_percent': relative_improvement,
        'num_runs': len(all_metrics),
        'num_proposed': len(proposed_runs),
        'num_baseline': len(baseline_runs),
    }
    
    return aggregated


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    
    print("=" * 80)
    print(f"Evaluating {len(run_ids)} runs")
    print(f"Run IDs: {run_ids}")
    print("=" * 80)
    
    # Fetch data for all runs
    all_run_data = {}
    all_metrics = {}
    
    for run_id in run_ids:
        print(f"\nProcessing run: {run_id}")
        
        # Fetch from WandB
        run_data = fetch_run_data(args.wandb_entity, args.wandb_project, run_id)
        all_run_data[run_id] = run_data
        
        # Export metrics
        metrics = export_run_metrics(run_data, args.results_dir)
        all_metrics[run_id] = metrics
        
        # Create per-run figures
        figures = create_run_figures(run_data, args.results_dir)
        for fig_path in figures:
            print(f"  Generated: {fig_path}")
    
    # Create comparison figures
    print("\n" + "=" * 80)
    print("Creating comparison visualizations")
    print("=" * 80)
    comparison_figures = create_comparison_figures(all_metrics, args.results_dir)
    for fig_path in comparison_figures:
        print(f"  Generated: {fig_path}")
    
    # Compute and export aggregated metrics
    print("\n" + "=" * 80)
    print("Computing aggregated metrics")
    print("=" * 80)
    aggregated = compute_aggregated_metrics(all_metrics)
    
    aggregated_path = os.path.join(args.results_dir, 'comparison', 'aggregated_metrics.json')
    os.makedirs(os.path.dirname(aggregated_path), exist_ok=True)
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nAggregated metrics saved to: {aggregated_path}")
    print(f"\nSummary:")
    print(f"  Best Proposed: {aggregated['best_proposed']['run_id']} = {aggregated['best_proposed']['accuracy']:.4f}")
    print(f"  Best Baseline: {aggregated['best_baseline']['run_id']} = {aggregated['best_baseline']['accuracy']:.4f}")
    print(f"  Gap: {aggregated['gap']:.4f} ({aggregated['relative_improvement_percent']:.2f}% improvement)")
    
    # Print all generated files
    print("\n" + "=" * 80)
    print("Generated Files:")
    print("=" * 80)
    for run_id in run_ids:
        print(f"{os.path.join(args.results_dir, run_id, 'metrics.json')}")
    for fig_path in comparison_figures:
        print(fig_path)
    print(aggregated_path)
    
    print("\n" + "=" * 80)
    print("Evaluation complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
