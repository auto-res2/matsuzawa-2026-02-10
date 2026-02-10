"""Data preprocessing for SVAMP dataset."""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm


class SVAMPDataset:
    """Handler for SVAMP arithmetic reasoning dataset."""
    
    def __init__(self, cache_dir: str = ".cache", random_seed: int = 42):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        
    def load_data(self, demo_pool_size: int = 500, test_size: int = 200) -> Tuple[List[Dict], List[Dict]]:
        """Load SVAMP data and split into demo pool and test set.
        
        Args:
            demo_pool_size: Number of examples for demo pool
            test_size: Number of examples for test set
            
        Returns:
            (demo_pool, test_set)
        """
        print(f"Loading SVAMP dataset from HuggingFace...")
        
        # Load SVAMP from HuggingFace
        try:
            dataset = load_dataset("ChilleD/SVAMP", cache_dir=str(self.cache_dir))
            train_data = dataset["train"]
            test_data = dataset["test"]
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Falling back to synthetic SVAMP-style data...")
            train_data, test_data = self._generate_synthetic_data(demo_pool_size + test_size)
            
        # Process and format data
        demo_pool = []
        test_set = []
        
        # Build demo pool from training data
        for i, example in enumerate(train_data):
            if i >= demo_pool_size:
                break
            demo_pool.append(self._format_example(example, f"demo_{i}"))
            
        # Build test set
        for i, example in enumerate(test_data):
            if i >= test_size:
                break
            test_set.append(self._format_example(example, f"test_{i}"))
            
        # If test set is too small, use additional training examples
        if len(test_set) < test_size and len(train_data) > demo_pool_size:
            remaining = test_size - len(test_set)
            for i in range(remaining):
                idx = demo_pool_size + i
                if idx < len(train_data):
                    test_set.append(self._format_example(train_data[idx], f"test_{len(test_set)}"))
                    
        print(f"Loaded {len(demo_pool)} demo pool examples and {len(test_set)} test examples")
        return demo_pool, test_set
        
    def _format_example(self, example: Dict, example_id: str) -> Dict:
        """Format a single example into our standard format."""
        # Handle different possible field names in SVAMP
        if "Body" in example and "Question" in example:
            question = f"{example['Body']} {example['Question']}"
        elif "question" in example:
            question = example["question"]
        else:
            question = str(example.get("Body", "")) + " " + str(example.get("Question", ""))
            
        # Extract answer
        if "Answer" in example:
            answer = float(example["Answer"])
        elif "answer" in example:
            answer = float(example["answer"])
        else:
            answer = 0.0
            
        return {
            "id": example_id,
            "question": question.strip(),
            "answer": answer
        }
        
    def _generate_synthetic_data(self, total_size: int) -> Tuple[List[Dict], List[Dict]]:
        """Generate synthetic SVAMP-style arithmetic problems as fallback."""
        print(f"Generating {total_size} synthetic arithmetic problems...")
        
        train_size = int(total_size * 0.7)
        test_size = total_size - train_size
        
        def generate_problem(idx: int) -> Dict:
            """Generate a single synthetic problem."""
            templates = [
                ("Sam has {a} apples. He gives {b} to his friend. How many apples does Sam have now?", 
                 lambda a, b: a - b),
                ("There are {a} birds on a tree. {b} more birds join them. How many birds are on the tree now?",
                 lambda a, b: a + b),
                ("A box contains {a} pencils. If {b} pencils are removed, how many pencils remain?",
                 lambda a, b: a - b),
                ("Maria has {a} candies. She buys {b} more candies. How many candies does she have in total?",
                 lambda a, b: a + b),
                ("There are {a} students in a class. {b} students are absent. How many students are present?",
                 lambda a, b: a - b),
            ]
            
            template, compute = random.choice(templates)
            a = random.randint(5, 100)
            b = random.randint(1, min(a, 50))
            
            question = template.format(a=a, b=b)
            answer = compute(a, b)
            
            return {
                "Body": question.split(".")[0] + ".",
                "Question": " ".join(question.split(".")[1:]).strip(),
                "Answer": str(answer)
            }
            
        train_data = [generate_problem(i) for i in range(train_size)]
        test_data = [generate_problem(i) for i in range(test_size)]
        
        return train_data, test_data
        
    def cluster_questions(
        self, 
        questions: List[Dict], 
        num_clusters: int = 8,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> Dict[int, List[int]]:
        """Cluster questions using sentence embeddings.
        
        Args:
            questions: List of question dictionaries
            num_clusters: Number of clusters (k)
            embedding_model_name: SentenceTransformer model name
            
        Returns:
            Dictionary mapping cluster_id -> list of question indices
        """
        print(f"Clustering {len(questions)} questions into {num_clusters} clusters...")
        
        # Load embedding model
        embedding_model = SentenceTransformer(
            embedding_model_name, 
            cache_folder=str(self.cache_dir)
        )
        
        # Generate embeddings
        question_texts = [q["question"] for q in questions]
        embeddings = embedding_model.encode(
            question_texts, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group questions by cluster
        clusters = {i: [] for i in range(num_clusters)}
        for idx, cluster_id in enumerate(cluster_labels):
            clusters[cluster_id].append(idx)
            
        print(f"Cluster sizes: {[len(clusters[i]) for i in range(num_clusters)]}")
        return clusters


def extract_answer_from_text(text: str) -> float:
    """Extract numeric answer from generated text.
    
    Strategy: Find the last number in the text (typically the final answer).
    """
    # Find all numbers in the text (including decimals)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if not numbers:
        return float('nan')
    
    # Return the last number found
    try:
        return float(numbers[-1])
    except (ValueError, IndexError):
        return float('nan')


def compute_accuracy(predictions: List[float], targets: List[float], tolerance: float = 1e-3) -> float:
    """Compute accuracy of numeric predictions.
    
    Args:
        predictions: Predicted answers
        targets: Ground truth answers
        tolerance: Tolerance for floating-point comparison
        
    Returns:
        Accuracy as a float in [0, 1]
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have same length")
    
    correct = 0
    for pred, target in zip(predictions, targets):
        # Handle NaN predictions
        if pred != pred:  # NaN check
            continue
        # Check if prediction is within tolerance
        if abs(pred - target) < tolerance:
            correct += 1
            
    return correct / len(predictions) if len(predictions) > 0 else 0.0
