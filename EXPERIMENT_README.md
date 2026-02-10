# C3-AutoCoT Experiment

## Overview

This experiment implements **C3-AutoCoT** (Cycle-Consistent & Paraphrase-Invariant Reliability Auto-CoT), a method for detecting and filtering out "plausible but ungrounded" chain-of-thought demonstrations in few-shot learning.

### Key Innovation

C3-AutoCoT extends PIR-AutoCoT by adding a **cycle-consistency** check to ensure demonstrations are not only internally consistent and paraphrase-invariant, but also properly grounded to the original question semantics.

**Reliability Score**: `r = r_sc × r_pi × r_cc`

- **r_sc** (self-consistency): consistency across multiple samples
- **r_pi** (paraphrase invariance): consistency across paraphrases
- **r_cc** (cycle consistency): can we reconstruct the question from reasoning?

## Project Structure

```
.
├── config/
│   ├── config.yaml                          # Base Hydra config
│   └── runs/
│       ├── proposed-llama3-svamp.yaml       # C3-AutoCoT (proposed)
│       ├── comparative-0-llama3-svamp.yaml  # PIR-AutoCoT (no cycle-consistency)
│       └── comparative-1-llama3-svamp.yaml  # Standard Auto-CoT (no reliability)
├── src/
│   ├── __init__.py
│   ├── main.py          # Orchestration script
│   ├── train.py         # Training/evaluation logic
│   ├── evaluate.py      # Results aggregation and comparison
│   ├── preprocess.py    # Dataset handling and clustering
│   └── model.py         # LLM wrapper and reliability scoring
├── pyproject.toml       # Dependencies
└── test_*.py           # Validation scripts
```

## Quick Start

### Installation

```bash
uv pip install -e .
```

### Training

Run a single experiment:

```bash
# Sanity check mode (fast, small data)
uv run python -u -m src.main run=proposed-llama3-svamp results_dir=results --sanity_check

# Full training mode
uv run python -u -m src.main run=proposed-llama3-svamp results_dir=results --main
```

### Evaluation

Aggregate and compare multiple runs:

```bash
uv run python -m src.evaluate results_dir=results run_ids='["proposed-llama3-svamp", "comparative-0-llama3-svamp"]'
```

## Run Configurations

### Proposed Method: C3-AutoCoT

**File**: `config/runs/proposed-llama3-svamp.yaml`

- **Run ID**: `proposed-llama3-svamp`
- **Method**: C3-AutoCoT with full reliability scoring (r_sc × r_pi × r_cc)
- **Model**: LLaMA-3-8B-Instruct
- **Dataset**: SVAMP (500 demo pool, 200 test)
- **Hyperparameter Search**: Optuna (20 trials)

### Baseline 1: PIR-AutoCoT

**File**: `config/runs/comparative-0-llama3-svamp.yaml`

- **Run ID**: `comparative-0-llama3-svamp`
- **Method**: PIR-AutoCoT (r_sc × r_pi, no cycle consistency)
- **Model**: LLaMA-3-8B-Instruct
- **Dataset**: SVAMP

### Baseline 2: Standard Auto-CoT

**File**: `config/runs/comparative-1-llama3-svamp.yaml`

- **Run ID**: `comparative-1-llama3-svamp`
- **Method**: Standard Auto-CoT (random sampling, no reliability)
- **Model**: LLaMA-3-8B-Instruct
- **Dataset**: SVAMP

## Modes

### Sanity Check Mode (`--sanity_check`)

- **Purpose**: Quick validation that code works
- **Data**: 50 demo pool, 10 test examples
- **WandB**: Disabled
- **Optuna**: Disabled (0 trials)
- **Output**: `SANITY_VALIDATION: PASS/FAIL` line

### Pilot Mode (`--pilot`)

- **Purpose**: Small-scale test before full run
- **Data**: 100 demo pool, 50 test examples
- **WandB**: Disabled
- **Optuna**: 5 trials

### Main Mode (`--main`)

- **Purpose**: Full experiment
- **Data**: 500 demo pool, 200 test examples
- **WandB**: Online
- **Optuna**: Full trials (if enabled)

## Output Structure

```
results/
├── {run_id}/
│   ├── metrics.json           # Final metrics
│   ├── results.json           # Per-example predictions
│   ├── demonstrations.json    # Selected demonstrations
│   ├── metrics_bar.png        # Metrics visualization
│   └── history_plot.png       # Training history (if available)
└── comparison/
    ├── aggregated_metrics.json     # Cross-run comparison
    ├── accuracy_comparison.png     # Bar chart
    └── method_comparison.png       # Box plot
```

### aggregated_metrics.json Structure

```json
{
  "primary_metric": "accuracy",
  "metrics": {
    "proposed-llama3-svamp": {"accuracy": 0.85, ...},
    "comparative-0-llama3-svamp": {"accuracy": 0.78, ...}
  },
  "best_proposed": {
    "run_id": "proposed-llama3-svamp",
    "accuracy": 0.85
  },
  "best_baseline": {
    "run_id": "comparative-0-llama3-svamp",
    "accuracy": 0.78
  },
  "gap": 0.07,
  "summary": {
    "num_runs": 3,
    "mean_proposed_accuracy": 0.85,
    "mean_baseline_accuracy": 0.76
  }
}
```

## Sanity Validation

In `--sanity_check` mode, the system automatically validates:

1. **Sufficient steps**: At least 5 test examples evaluated
2. **Finite metrics**: All metrics are finite (no NaN/Inf)
3. **Non-zero accuracy**: Accuracy > 0 (model is learning)
4. **Metrics exist**: Required metrics are logged

Output format:
```
SANITY_VALIDATION: PASS
SANITY_VALIDATION_SUMMARY: {"steps":10, "accuracy":0.4, "num_test":10, "num_demos":8}
```

Or:
```
SANITY_VALIDATION: FAIL reason=zero_accuracy
SANITY_VALIDATION_SUMMARY: {"steps":10, "accuracy":0.0, "num_test":10, "num_demos":8}
```

## WandB Integration

### Training (src/train.py)

```python
wandb.init(
    entity=cfg.wandb.entity,      # "airas"
    project=cfg.wandb.project,    # "2026-02-10-test"
    id=cfg.run.run_id,
    config=OmegaConf.to_container(cfg, resolve=True),
    resume="allow",
    mode=cfg.wandb.mode           # "online" or "disabled"
)

wandb.log(metrics)
wandb.summary.update(metrics)
```

### Evaluation (src/evaluate.py)

```python
api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")
history = run.history()
summary = dict(run.summary)
config = dict(run.config)
```

## Dataset: SVAMP

- **Source**: HuggingFace `ChilleD/SVAMP`
- **Task**: Elementary arithmetic word problems
- **Format**: 
  - Input: Word problem text
  - Output: Numeric answer
- **Splits**:
  - Demo pool: First 500 training examples
  - Test set: 200 test examples

### Clustering

- **Method**: K-means on sentence embeddings
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **K**: 8 clusters
- **Selection**: 1 demonstration per cluster (highest reliability)

## Model: LLaMA-3-8B-Instruct

- **Name**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Inference**: Greedy decoding (temperature=0.0)
- **Max tokens**: 256
- **Device**: CUDA
- **Quantization**: Optional 8-bit loading

## Hyperparameter Optimization (Optuna)

When enabled, Optuna searches over:

- `reliability_threshold`: [0.4, 0.8]
- `self_consistency_samples`: [3, 7]

**Objective**: Maximize accuracy on validation split

**Study configuration**:
- Sampler: TPESampler
- Pruner: MedianPruner
- Trials: 20 (main), 0 (sanity check)

## Metrics

### Primary Metric: Accuracy

Proportion of test questions with correct final numeric answer.

**Extraction**: Last number in generated output
**Comparison**: Absolute difference < 1e-3

### Additional Metrics

- `num_test`: Number of test examples
- `num_demos`: Number of selected demonstrations
- `reliability` (per demo): Composite reliability score

## Testing

### Structure Test

```bash
python test_structure.py
```

Validates:
- All required files exist
- YAML syntax is valid
- Python syntax is valid
- Basic imports work

### Sanity Test

```bash
python test_sanity.py
```

Validates:
- Config structure and naming conventions
- CLI interface implementation
- Sanity validation logic
- WandB integration
- Results structure
- Data leakage prevention

## CLI Reference

### Training Command

```bash
uv run python -u -m src.main \
  run={run_id} \
  results_dir={path} \
  {--sanity_check|--pilot|--main}
```

**Parameters**:
- `run={run_id}`: Select run config from `config/runs/`
- `results_dir={path}`: Output directory for results
- `--sanity_check`: Fast validation mode
- `--pilot`: Medium-scale test mode
- `--main`: Full experiment mode

**Examples**:

```bash
# Sanity check
uv run python -u -m src.main run=proposed-llama3-svamp results_dir=results --sanity_check

# Pilot run
uv run python -u -m src.main run=proposed-llama3-svamp results_dir=results --pilot

# Full run
uv run python -u -m src.main run=proposed-llama3-svamp results_dir=results --main
```

### Evaluation Command

```bash
uv run python -m src.evaluate \
  results_dir={path} \
  run_ids='["run-1", "run-2", ...]' \
  [--entity {wandb_entity}] \
  [--project {wandb_project}] \
  [--skip-wandb]
```

**Parameters**:
- `results_dir`: Results directory
- `run_ids`: JSON list of run IDs to compare
- `--entity`: WandB entity (default: "airas")
- `--project`: WandB project (default: "2026-02-10-test")
- `--skip-wandb`: Use only local metrics

**Example**:

```bash
uv run python -m src.evaluate \
  results_dir=results \
  run_ids='["proposed-llama3-svamp", "comparative-0-llama3-svamp", "comparative-1-llama3-svamp"]'
```

## Implementation Details

### Demo Selection Algorithm

1. Load demo pool (500 examples from SVAMP training set)
2. Cluster questions using sentence embeddings (k=8)
3. For each cluster:
   - Shuffle candidates
   - For each candidate:
     - Generate CoT reasoning
     - Check answer correctness
     - Compute reliability score
     - If score ≥ threshold: select and break
   - Fallback: select first correct candidate
4. Return selected demonstrations (1 per cluster = 8 total)

### Reliability Scoring

#### Self-Consistency (r_sc)

```python
def compute_self_consistency(question, reference_answer, num_samples=5, temp=0.7):
    consistent = 0
    for _ in range(num_samples):
        prediction = model.generate(question, temperature=temp)
        if matches(prediction, reference_answer):
            consistent += 1
    return consistent / num_samples
```

#### Paraphrase Invariance (r_pi)

```python
def compute_paraphrase_invariance(question, reference_answer, num_paraphrases=3):
    consistent = 0
    for _ in range(num_paraphrases):
        paraphrase = model.paraphrase(question)
        prediction = model.generate(paraphrase)
        if matches(prediction, reference_answer):
            consistent += 1
    return consistent / num_paraphrases
```

#### Cycle Consistency (r_cc)

```python
def compute_cycle_consistency(question, reasoning):
    reconstructed = model.reconstruct_question(reasoning)
    similarity = jaccard_similarity(question, reconstructed)
    return similarity
```

## Dependencies

All dependencies are specified in `pyproject.toml`:

- **Core ML**: torch, transformers, accelerate
- **Config**: hydra-core, omegaconf
- **Tracking**: wandb
- **Data**: datasets, sentence-transformers
- **Optimization**: optuna
- **Visualization**: matplotlib, seaborn
- **Utilities**: numpy, scipy, pandas, scikit-learn, tqdm

## Troubleshooting

### Import Errors

Make sure dependencies are installed:
```bash
uv pip install -e .
```

### CUDA Out of Memory

Enable 8-bit quantization in config:
```yaml
model:
  load_in_8bit: true
```

### WandB Authentication

Login before running:
```bash
wandb login
```

Or skip WandB in sanity check:
```bash
uv run python -u -m src.main run=proposed-llama3-svamp results_dir=results --sanity_check
```

### Hydra Config Errors

Check config syntax:
```bash
python test_sanity.py
```

View resolved config:
```bash
uv run python -c "from omegaconf import OmegaConf; print(OmegaConf.to_yaml(OmegaConf.load('config/config.yaml')))"
```

## License

See LICENSE file.

## Citation

If you use this code, please cite:

```bibtex
@misc{c3autocot2026,
  title={Cycle-Consistent \& Paraphrase-Invariant Reliability Auto-CoT},
  author={Your Name},
  year={2026}
}
```
