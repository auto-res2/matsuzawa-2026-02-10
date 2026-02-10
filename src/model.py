"""
Model implementation for C3-AutoCoT and PIR-AutoCoT.
"""
import os
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import numpy as np
from collections import Counter

from src.preprocess import (
    extract_answer_from_text,
    compare_answers,
    normalize_answer,
    generate_paraphrase_prompt,
    generate_reconstruction_prompt,
    format_cot_prompt
)


class AutoCoTModel:
    """
    Base class for Auto-CoT models supporting both causal LM and seq2seq models.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        cache_dir: str = ".cache/",
        device: str = "cuda",
        max_new_tokens: int = 512,
        load_in_8bit: bool = False
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Detect model type
        self.is_seq2seq = any(name in model_name.lower() for name in ['t5', 'bart', 'pegasus'])
        
        # Load tokenizer and model
        print(f"Loading model {model_name} (seq2seq={self.is_seq2seq})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            padding_side='left' if not self.is_seq2seq else 'right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            "cache_dir": cache_dir,
        }
        
        # Use CPU if CUDA not available
        if not torch.cuda.is_available():
            device = "cpu"
            self.device = device
            print("CUDA not available, using CPU")
        else:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        
        if load_in_8bit and torch.cuda.is_available():
            model_kwargs["load_in_8bit"] = True
        
        # Load appropriate model type
        if self.is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs and device != "auto":
            self.model = self.model.to(device)
        
        self.model.eval()
        print("Model loaded successfully.")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from a prompt.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move inputs to device
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        
        if temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.9
            })
        else:
            generation_kwargs["do_sample"] = False
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            if self.is_seq2seq:
                # For seq2seq models, decode the entire output
                text = self.tokenizer.decode(output, skip_special_tokens=True)
            else:
                # For causal LM, skip the input tokens
                generated = output[inputs['input_ids'].shape[1]:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def generate_cot_rationale(
        self,
        question: str,
        temperature: float = 0.7
    ) -> Tuple[str, str]:
        """
        Generate a chain-of-thought rationale and answer for a question.
        
        Returns:
            rationale: The reasoning steps
            answer: The final numeric answer
        """
        prompt = f"""Solve the following math word problem step by step. Show your reasoning and provide the final numeric answer.

Question: {question}
Reasoning:"""
        
        output = self.generate(prompt, temperature=temperature, num_return_sequences=1)[0]
        
        # Split into rationale and extract answer
        lines = output.strip().split('\n')
        
        # Look for explicit answer line or extract from end
        answer = extract_answer_from_text(output)
        rationale = output.strip()
        
        return rationale, answer
    
    def compute_self_consistency(
        self,
        question: str,
        num_samples: int = 5,
        temperature: float = 0.7
    ) -> float:
        """
        Compute self-consistency score by sampling multiple times.
        Returns the proportion of samples that agree with the majority answer.
        """
        answers = []
        
        for _ in range(num_samples):
            _, answer = self.generate_cot_rationale(question, temperature=temperature)
            if answer:
                answers.append(normalize_answer(answer))
        
        if not answers:
            return 0.0
        
        # Find majority answer
        answer_counts = Counter(answers)
        majority_count = answer_counts.most_common(1)[0][1]
        
        # Self-consistency score is the proportion agreeing with majority
        return majority_count / len(answers)
    
    def compute_paraphrase_invariance(
        self,
        question: str,
        original_answer: str,
        num_paraphrases: int = 3,
        temperature: float = 0.5
    ) -> float:
        """
        Compute paraphrase invariance by checking consistency across paraphrases.
        Returns the proportion of paraphrases that yield the same answer.
        """
        consistent_count = 0
        
        for _ in range(num_paraphrases):
            # Generate paraphrase
            paraphrase_prompt = generate_paraphrase_prompt(question)
            paraphrased = self.generate(paraphrase_prompt, temperature=temperature)[0]
            
            # Get answer for paraphrased question
            _, para_answer = self.generate_cot_rationale(paraphrased, temperature=0.0)
            
            # Check if answers match
            if compare_answers(original_answer, para_answer):
                consistent_count += 1
        
        return consistent_count / num_paraphrases
    
    def compute_cycle_consistency(
        self,
        question: str,
        rationale: str,
        answer: str,
        temperature: float = 0.5
    ) -> float:
        """
        Compute cycle consistency by reconstructing the question from the rationale.
        Returns a similarity score between original and reconstructed question.
        """
        # Reconstruct question from rationale
        reconstruct_prompt = generate_reconstruction_prompt(rationale, answer)
        reconstructed = self.generate(reconstruct_prompt, temperature=temperature)[0]
        
        # Compute semantic similarity using simple word overlap
        # In a production system, you'd use embeddings here
        q1_words = set(question.lower().split())
        q2_words = set(reconstructed.lower().split())
        
        if not q1_words or not q2_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(q1_words & q2_words)
        union = len(q1_words | q2_words)
        
        return intersection / union if union > 0 else 0.0


class C3AutoCoT(AutoCoTModel):
    """
    C3-AutoCoT: Cycle-Consistent & Paraphrase-Invariant Reliability Auto-CoT.
    """
    
    def select_demonstrations(
        self,
        demo_pool: List[Dict],
        cluster_labels: np.ndarray,
        num_clusters: int,
        reliability_threshold: float,
        self_consistency_config: Dict,
        paraphrase_invariance_config: Dict,
        cycle_consistency_config: Dict,
        max_candidates_per_cluster: int = None,
    ) -> List[Dict]:
        """
        Select demonstrations using C3-AutoCoT reliability scoring.
        """
        selected_demos = []
        
        for cluster_id in range(num_clusters):
            # Get questions in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Limit candidates if specified
            if max_candidates_per_cluster is not None:
                cluster_indices = cluster_indices[:max_candidates_per_cluster]
            
            best_demo = None
            best_reliability = -1
            
            for idx in cluster_indices:
                question = demo_pool[idx]['question']
                
                # Generate CoT rationale
                rationale, answer = self.generate_cot_rationale(
                    question,
                    temperature=self_consistency_config['temperature']
                )
                
                if not answer:
                    continue
                
                # Compute reliability scores
                r_sc = self.compute_self_consistency(
                    question,
                    num_samples=self_consistency_config['num_samples'],
                    temperature=self_consistency_config['temperature']
                )
                
                r_pi = self.compute_paraphrase_invariance(
                    question,
                    answer,
                    num_paraphrases=paraphrase_invariance_config['num_paraphrases'],
                    temperature=paraphrase_invariance_config['temperature']
                )
                
                r_cc = self.compute_cycle_consistency(
                    question,
                    rationale,
                    answer,
                    temperature=cycle_consistency_config['temperature']
                ) if cycle_consistency_config['enabled'] else 1.0
                
                # Combined reliability
                reliability = r_sc * r_pi * r_cc
                
                print(f"Cluster {cluster_id}, Q: {question[:50]}... | r_sc={r_sc:.3f}, r_pi={r_pi:.3f}, r_cc={r_cc:.3f}, r={reliability:.3f}")
                
                if reliability >= reliability_threshold and reliability > best_reliability:
                    best_reliability = reliability
                    best_demo = {
                        'question': question,
                        'rationale': rationale,
                        'answer': answer,
                        'reliability': reliability,
                        'r_sc': r_sc,
                        'r_pi': r_pi,
                        'r_cc': r_cc
                    }
            
            if best_demo:
                selected_demos.append(best_demo)
                print(f"✓ Cluster {cluster_id}: Selected demo with reliability {best_reliability:.3f}")
            else:
                print(f"✗ Cluster {cluster_id}: No demo met threshold {reliability_threshold}")
        
        return selected_demos


class PIRAutoCoT(AutoCoTModel):
    """
    PIR-AutoCoT: Paraphrase-Invariant Reliability Auto-CoT (baseline without cycle consistency).
    """
    
    def select_demonstrations(
        self,
        demo_pool: List[Dict],
        cluster_labels: np.ndarray,
        num_clusters: int,
        reliability_threshold: float,
        self_consistency_config: Dict,
        paraphrase_invariance_config: Dict,
        cycle_consistency_config: Dict,
        max_candidates_per_cluster: int = None,
    ) -> List[Dict]:
        """
        Select demonstrations using PIR-AutoCoT reliability scoring (no cycle consistency).
        """
        selected_demos = []
        
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Limit candidates if specified
            if max_candidates_per_cluster is not None:
                cluster_indices = cluster_indices[:max_candidates_per_cluster]
            
            best_demo = None
            best_reliability = -1
            
            for idx in cluster_indices:
                question = demo_pool[idx]['question']
                
                # Generate CoT rationale
                rationale, answer = self.generate_cot_rationale(
                    question,
                    temperature=self_consistency_config['temperature']
                )
                
                if not answer:
                    continue
                
                # Compute reliability scores (no cycle consistency)
                r_sc = self.compute_self_consistency(
                    question,
                    num_samples=self_consistency_config['num_samples'],
                    temperature=self_consistency_config['temperature']
                )
                
                r_pi = self.compute_paraphrase_invariance(
                    question,
                    answer,
                    num_paraphrases=paraphrase_invariance_config['num_paraphrases'],
                    temperature=paraphrase_invariance_config['temperature']
                )
                
                # Combined reliability (no r_cc term)
                reliability = r_sc * r_pi
                
                print(f"Cluster {cluster_id}, Q: {question[:50]}... | r_sc={r_sc:.3f}, r_pi={r_pi:.3f}, r={reliability:.3f}")
                
                if reliability >= reliability_threshold and reliability > best_reliability:
                    best_reliability = reliability
                    best_demo = {
                        'question': question,
                        'rationale': rationale,
                        'answer': answer,
                        'reliability': reliability,
                        'r_sc': r_sc,
                        'r_pi': r_pi,
                        'r_cc': 1.0  # Not used but included for consistency
                    }
            
            if best_demo:
                selected_demos.append(best_demo)
                print(f"✓ Cluster {cluster_id}: Selected demo with reliability {best_reliability:.3f}")
            else:
                print(f"✗ Cluster {cluster_id}: No demo met threshold {reliability_threshold}")
        
        return selected_demos


def create_model(method_type: str, model_config: Dict) -> AutoCoTModel:
    """
    Factory function to create the appropriate model.
    """
    if method_type == "c3_autocot":
        return C3AutoCoT(**model_config)
    elif method_type == "pir_autocot":
        return PIRAutoCoT(**model_config)
    else:
        raise ValueError(f"Unknown method type: {method_type}")
