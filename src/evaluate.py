"""Evaluation and comparison script for C3-AutoCoT experiments."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb


def load_wandb_run(run_id: str, entity: str, project: str) -> Dict:
    """Load run data from WandB API.
    
    Args:
        run_id: Run ID
        entity: WandB entity
        project: WandB project
        
    Returns:
        Dictionary with history, summary, and config
    """
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get history (all logged metrics over time)
        history = run.history()
        
        # Get summary (final metrics)
        summary = dict(run.summary)
        
        # Get config
        config = dict(run.config)
        
        return {
            "history": history,
            "summary": summary,
            "config": config,
            "run_id": run_id
        }
    except Exception as e:
        print(f"Warning: Could not load WandB run {run_id}: {e}")
        return None


def load_local_metrics(results_dir: Path, run_id: str) -> Dict:
    """Load metrics from local results directory.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        
    Returns:
        Metrics dictionary
    """
    metrics_path = results_dir / run_id / "metrics.json"
    
    if not metrics_path.exists():
        print(f"Warning: No metrics found at {metrics_path}")
        return {}
        
    with open(metrics_path, "r") as f:
        return json.load(f)


def export_per_run_metrics(results_dir: Path, run_id: str, metrics: Dict):
    """Export per-run metrics to JSON.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        metrics: Metrics dictionary
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = run_dir / "metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Exported metrics: {output_path}")


def create_per_run_figures(results_dir: Path, run_id: str, metrics: Dict, history: pd.DataFrame = None):
    """Create per-run visualization figures.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        metrics: Metrics dictionary
        history: Optional history dataframe
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Metrics bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metric_names = [k for k in metrics.keys() if isinstance(metrics[k], (int, float))]
    metric_values = [metrics[k] for k in metric_names]
    
    ax.bar(metric_names, metric_values)
    ax.set_ylabel("Value")
    ax.set_title(f"Metrics for {run_id}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    output_path = run_dir / "metrics_bar.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created figure: {output_path}")
    
    # Figure 2: History plot (if available)
    if history is not None and not history.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for col in history.columns:
            if col not in ["_step", "_timestamp", "_runtime"]:
                ax.plot(history["_step"], history[col], label=col, marker="o")
                
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(f"Training History: {run_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = run_dir / "history_plot.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Created figure: {output_path}")


def create_comparison_figures(results_dir: Path, all_metrics: Dict[str, Dict]):
    """Create comparison figures across runs.
    
    Args:
        results_dir: Results directory
        all_metrics: Dictionary mapping run_id -> metrics
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract accuracy for each run
    run_ids = list(all_metrics.keys())
    accuracies = [all_metrics[rid].get("accuracy", 0.0) for rid in run_ids]
    
    # Figure 1: Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids]
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors)
    
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy Comparison")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
                
    plt.tight_layout()
    output_path = comparison_dir / "accuracy_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created figure: {output_path}")
    
    # Figure 2: Method type comparison (proposed vs baselines)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    proposed_accs = [all_metrics[rid].get("accuracy", 0.0) 
                     for rid in run_ids if "proposed" in rid]
    baseline_accs = [all_metrics[rid].get("accuracy", 0.0) 
                     for rid in run_ids if "comparative" in rid]
    
    box_data = []
    labels = []
    if proposed_accs:
        box_data.append(proposed_accs)
        labels.append("Proposed")
    if baseline_accs:
        box_data.append(baseline_accs)
        labels.append("Baselines")
        
    if box_data:
        ax.boxplot(box_data, labels=labels)
        ax.set_ylabel("Accuracy")
        ax.set_title("Proposed vs. Baselines")
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        output_path = comparison_dir / "method_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Created figure: {output_path}")


def export_aggregated_metrics(results_dir: Path, all_metrics: Dict[str, Dict]):
    """Export aggregated metrics across all runs.
    
    Args:
        results_dir: Results directory
        all_metrics: Dictionary mapping run_id -> metrics
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate proposed and baselines
    proposed_metrics = {rid: m for rid, m in all_metrics.items() if "proposed" in rid}
    baseline_metrics = {rid: m for rid, m in all_metrics.items() if "comparative" in rid}
    
    # Find best runs
    best_proposed = None
    best_proposed_acc = 0.0
    for rid, metrics in proposed_metrics.items():
        acc = metrics.get("accuracy", 0.0)
        if acc > best_proposed_acc:
            best_proposed_acc = acc
            best_proposed = rid
            
    best_baseline = None
    best_baseline_acc = 0.0
    for rid, metrics in baseline_metrics.items():
        acc = metrics.get("accuracy", 0.0)
        if acc > best_baseline_acc:
            best_baseline_acc = acc
            best_baseline = rid
            
    # Compute gap
    gap = best_proposed_acc - best_baseline_acc if best_proposed and best_baseline else 0.0
    
    # Build aggregated metrics
    aggregated = {
        "primary_metric": "accuracy",
        "metrics": all_metrics,
        "best_proposed": {
            "run_id": best_proposed,
            "accuracy": best_proposed_acc
        },
        "best_baseline": {
            "run_id": best_baseline,
            "accuracy": best_baseline_acc
        },
        "gap": gap,
        "summary": {
            "num_runs": len(all_metrics),
            "num_proposed": len(proposed_metrics),
            "num_baselines": len(baseline_metrics),
            "mean_proposed_accuracy": np.mean([m.get("accuracy", 0.0) for m in proposed_metrics.values()]) if proposed_metrics else 0.0,
            "mean_baseline_accuracy": np.mean([m.get("accuracy", 0.0) for m in baseline_metrics.values()]) if baseline_metrics else 0.0
        }
    }
    
    output_path = comparison_dir / "aggregated_metrics.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
        
    print(f"Exported aggregated metrics: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Best Proposed: {best_proposed} (accuracy={best_proposed_acc:.4f})")
    print(f"Best Baseline: {best_baseline} (accuracy={best_baseline_acc:.4f})")
    print(f"Gap: {gap:.4f}")
    print("="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate C3-AutoCoT experiments")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("run_ids", type=str, help="JSON string list of run IDs")
    parser.add_argument("--entity", type=str, default="airas", help="WandB entity")
    parser.add_argument("--project", type=str, default="2026-02-10-test", help="WandB project")
    parser.add_argument("--skip-wandb", action="store_true", help="Skip WandB API calls")
    
    args = parser.parse_args()
    
    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    results_dir = Path(args.results_dir)
    
    print(f"Evaluating {len(run_ids)} runs...")
    print(f"Results directory: {results_dir}")
    
    # Collect metrics for all runs
    all_metrics = {}
    
    for run_id in run_ids:
        print(f"\nProcessing run: {run_id}")
        
        # Try to load from WandB first
        wandb_data = None
        if not args.skip_wandb:
            wandb_data = load_wandb_run(run_id, args.entity, args.project)
            
        # Load local metrics as fallback
        local_metrics = load_local_metrics(results_dir, run_id)
        
        # Merge metrics (prefer WandB summary, fallback to local)
        if wandb_data and wandb_data["summary"]:
            metrics = wandb_data["summary"]
        else:
            metrics = local_metrics
            
        # Export per-run metrics
        export_per_run_metrics(results_dir, run_id, metrics)
        
        # Create per-run figures
        history = wandb_data["history"] if wandb_data else None
        create_per_run_figures(results_dir, run_id, metrics, history)
        
        all_metrics[run_id] = metrics
        
    # Create comparison figures
    if len(all_metrics) > 1:
        print("\nCreating comparison figures...")
        create_comparison_figures(results_dir, all_metrics)
        
    # Export aggregated metrics
    print("\nExporting aggregated metrics...")
    export_aggregated_metrics(results_dir, all_metrics)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
