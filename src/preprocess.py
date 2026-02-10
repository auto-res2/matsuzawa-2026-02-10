"""
Data preprocessing for SVAMP dataset.
"""
import os
import json
import re
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def load_svamp_dataset(cache_dir: str = ".cache/", seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Load SVAMP dataset from HuggingFace.
    
    Returns:
        train_data: List of training examples
        test_data: List of test examples
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load SVAMP dataset
    dataset = load_dataset("ChilleD/SVAMP", cache_dir=cache_dir)
    
    train_data = []
    test_data = []
    
    # SVAMP has train split - we'll use it for demo pool and part for testing
    if "train" in dataset:
        all_data = list(dataset["train"])
        np.random.seed(seed)
        np.random.shuffle(all_data)
        
        # Use first part for training (demo pool), rest for testing
        train_data = all_data[:700]  # More than needed for 500 demo pool
        test_data = all_data[700:900]  # 200 for testing
    
    # Convert to our format
    train_examples = []
    for item in train_data:
        train_examples.append({
            "question": item["Question"],
            "answer": str(item["Answer"]),
            "equation": item.get("Equation", ""),
            "type": item.get("Type", "")
        })
    
    test_examples = []
    for item in test_data:
        test_examples.append({
            "question": item["Question"],
            "answer": str(item["Answer"]),
            "equation": item.get("Equation", ""),
            "type": item.get("Type", "")
        })
    
    return train_examples, test_examples


def cluster_questions(
    questions: List[str],
    num_clusters: int = 8,
    cache_dir: str = ".cache/"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster questions using SentenceTransformer embeddings and k-means.
    
    Args:
        questions: List of question strings
        num_clusters: Number of clusters
        cache_dir: Cache directory for models
        
    Returns:
        cluster_labels: Array of cluster assignments
        embeddings: Question embeddings
    """
    # Load sentence transformer
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
    
    # Encode questions
    embeddings = model.encode(questions, show_progress_bar=False)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels, embeddings


def extract_answer_from_text(text: str) -> str:
    """
    Extract the final numeric answer from model output.
    Looks for the last number in the text.
    """
    # Find all numbers (including decimals)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if numbers:
        # Return the last number found
        return numbers[-1]
    
    return ""


def format_cot_prompt(question: str, demos: List[Dict[str, str]]) -> str:
    """
    Format a chain-of-thought prompt with demonstrations.
    
    Args:
        question: The question to answer
        demos: List of demonstration examples with 'question', 'rationale', 'answer'
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        "Solve the following math word problems step by step. Show your reasoning and provide the final numeric answer.\n"
    ]
    
    # Add demonstrations
    for i, demo in enumerate(demos, 1):
        prompt_parts.append(f"\nExample {i}:")
        prompt_parts.append(f"Question: {demo['question']}")
        if 'rationale' in demo and demo['rationale']:
            prompt_parts.append(f"Reasoning: {demo['rationale']}")
        prompt_parts.append(f"Answer: {demo['answer']}\n")
    
    # Add test question
    prompt_parts.append(f"\nNow solve this problem:")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append(f"Reasoning:")
    
    return "\n".join(prompt_parts)


def generate_paraphrase_prompt(question: str) -> str:
    """
    Generate a prompt to paraphrase a question.
    """
    return f"""Paraphrase the following math word problem. Keep the same numbers and mathematical relationships, but rewrite it in different words.

Original: {question}

Paraphrased:"""


def generate_reconstruction_prompt(rationale: str, answer: str) -> str:
    """
    Generate a prompt to reconstruct the original question from a CoT rationale.
    This is used for cycle consistency checking.
    """
    return f"""Given the following reasoning and answer, reconstruct the original math word problem that was being solved.

Reasoning: {rationale}
Answer: {answer}

Original question:"""


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    """
    # Remove whitespace and convert to lowercase
    answer = answer.strip().lower()
    
    # Extract just the numeric part
    numbers = re.findall(r'-?\d+\.?\d*', answer)
    if numbers:
        return numbers[0]
    
    return answer


def compare_answers(answer1: str, answer2: str, tolerance: float = 1e-5) -> bool:
    """
    Compare two answers for equality (with tolerance for floating point).
    """
    try:
        a1 = float(normalize_answer(answer1))
        a2 = float(normalize_answer(answer2))
        return abs(a1 - a2) < tolerance
    except (ValueError, TypeError):
        # Fallback to string comparison
        return normalize_answer(answer1) == normalize_answer(answer2)


def save_demo_pool(demo_pool: List[Dict], output_path: str):
    """
    Save the selected demo pool to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(demo_pool, f, indent=2)


def load_demo_pool(input_path: str) -> List[Dict]:
    """
    Load a demo pool from a JSON file.
    """
    with open(input_path, 'r') as f:
        return json.load(f)
