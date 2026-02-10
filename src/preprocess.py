"""
SVAMP Dataset Preprocessing and Clustering
"""
import os
import json
import re
import numpy as np
from typing import List, Dict, Tuple
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def load_svamp_dataset(cache_dir: str = ".cache/datasets") -> Tuple[List[Dict], List[Dict]]:
    """
    Load SVAMP dataset from HuggingFace.
    Returns (train_examples, test_examples)
    """
    dataset = load_dataset("ChilleD/SVAMP", cache_dir=cache_dir)
    
    train_data = []
    for item in dataset['train']:
        train_data.append({
            'question': item['Question'],
            'equation': item.get('Equation', ''),
            'answer': float(item['Answer']),
            'body': item.get('Body', ''),
            'type': item.get('Type', '')
        })
    
    test_data = []
    for item in dataset['test']:
        test_data.append({
            'question': item['Question'],
            'equation': item.get('Equation', ''),
            'answer': float(item['Answer']),
            'body': item.get('Body', ''),
            'type': item.get('Type', '')
        })
    
    return train_data, test_data


def extract_numeric_answer(text: str) -> float:
    """
    Extract the last numeric value from generated text.
    """
    # Find all numbers (including decimals and negatives)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def cluster_questions(
    questions: List[str],
    num_clusters: int,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str = ".cache/models"
) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Cluster questions using sentence embeddings and k-means.
    Returns (cluster_labels, embedding_model)
    """
    # Load sentence transformer
    model = SentenceTransformer(embedding_model_name, cache_folder=cache_dir)
    
    # Embed questions
    embeddings = model.encode(questions, show_progress_bar=False)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels, model


def prepare_demo_pool(
    train_data: List[Dict],
    pool_size: int,
    num_clusters: int,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str = ".cache/models"
) -> Dict[int, List[Dict]]:
    """
    Prepare the demo pool: first pool_size training examples, clustered.
    Returns dict mapping cluster_id -> list of examples
    """
    # Take first pool_size examples
    demo_pool = train_data[:pool_size]
    questions = [ex['question'] for ex in demo_pool]
    
    # Cluster
    cluster_labels, _ = cluster_questions(
        questions, num_clusters, embedding_model_name, cache_dir
    )
    
    # Group by cluster
    clusters = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(demo_pool[idx])
    
    return clusters


def generate_paraphrases(question: str, num_samples: int = 3) -> List[str]:
    """
    Generate simple paraphrases of a question.
    For lightweight implementation, we use rule-based transformations.
    In a full system, this would use the LLM.
    """
    paraphrases = [question]  # Original is always included
    
    # Simple transformations
    # 1. Reorder clauses if there's a conjunction
    if ' and ' in question.lower():
        parts = question.split(' and ')
        if len(parts) == 2:
            paraphrases.append(f"{parts[1].strip()} and {parts[0].strip()}")
    
    # 2. Add/remove question mark
    if question.endswith('?'):
        paraphrases.append(question[:-1] + '.')
    else:
        paraphrases.append(question + '?')
    
    # Return up to num_samples unique paraphrases
    paraphrases = list(set(paraphrases))[:num_samples]
    return paraphrases


def compute_self_consistency_score(outputs: List[str]) -> float:
    """
    Compute r_sc: agreement rate among multiple samples.
    """
    if not outputs:
        return 0.0
    
    answers = [extract_numeric_answer(out) for out in outputs]
    answers = [a for a in answers if a is not None]
    
    if len(answers) <= 1:
        return 0.0
    
    # Count most common answer
    from collections import Counter
    counts = Counter(answers)
    most_common_count = counts.most_common(1)[0][1]
    
    return most_common_count / len(answers)


def compute_paraphrase_invariance_score(
    original_outputs: List[str],
    paraphrase_outputs: List[List[str]]
) -> float:
    """
    Compute r_pi: agreement between original and paraphrase samples.
    """
    if not original_outputs or not paraphrase_outputs:
        return 0.0
    
    # Extract answers
    original_answers = [extract_numeric_answer(out) for out in original_outputs]
    original_answers = [a for a in original_answers if a is not None]
    
    if not original_answers:
        return 0.0
    
    # Majority answer from original
    from collections import Counter
    original_majority = Counter(original_answers).most_common(1)[0][0]
    
    # Check agreement with paraphrases
    agreements = []
    for para_outs in paraphrase_outputs:
        para_answers = [extract_numeric_answer(out) for out in para_outs]
        para_answers = [a for a in para_answers if a is not None]
        if para_answers:
            para_majority = Counter(para_answers).most_common(1)[0][0]
            agreements.append(1.0 if abs(para_majority - original_majority) < 1e-6 else 0.0)
    
    return np.mean(agreements) if agreements else 0.0


def compute_cycle_consistency_score(
    question: str,
    rationale: str,
    reconstructed_question: str
) -> float:
    """
    Compute r_cc: semantic similarity between original and reconstructed question.
    Uses simple token overlap for lightweight implementation.
    """
    # Tokenize
    def tokenize(text):
        return set(re.findall(r'\w+', text.lower()))
    
    q_tokens = tokenize(question)
    r_tokens = tokenize(reconstructed_question)
    
    if not q_tokens:
        return 0.0
    
    # Jaccard similarity
    intersection = len(q_tokens & r_tokens)
    union = len(q_tokens | r_tokens)
    
    return intersection / union if union > 0 else 0.0


def save_demonstrations(demos: List[Dict], output_path: str):
    """Save selected demonstrations to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(demos, f, indent=2)


def load_demonstrations(input_path: str) -> List[Dict]:
    """Load demonstrations from JSON."""
    with open(input_path, 'r') as f:
        return json.load(f)
