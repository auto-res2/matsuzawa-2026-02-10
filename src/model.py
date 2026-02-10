"""Model wrappers and reliability scoring for C3-AutoCoT."""

import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMWrapper:
    """Wrapper for language model inference."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: str = ".cache",
        load_in_8bit: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.0
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        model_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not load_in_8bit:
            self.model = self.model.to(device)
            
        self.model.eval()
        print(f"Model loaded on {device}")
        
    def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (None = use default)
            max_new_tokens: Max tokens to generate (None = use default)
            
        Returns:
            Generated text
        """
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
            
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            if temperature == 0.0:
                # Greedy decoding
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                # Sampling
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
        
    def generate_cot(self, question: str, demos: List[Dict] = None) -> str:
        """Generate chain-of-thought reasoning for a question.
        
        Args:
            question: Question to answer
            demos: Optional list of demonstration examples
            
        Returns:
            Generated CoT reasoning + answer
        """
        prompt = self._build_cot_prompt(question, demos)
        return self.generate(prompt)
        
    def _build_cot_prompt(self, question: str, demos: List[Dict] = None) -> str:
        """Build a CoT prompt with optional demonstrations."""
        prompt_parts = []
        
        # Add system instruction
        prompt_parts.append(
            "Solve the following arithmetic word problems step by step. "
            "Show your reasoning and provide the final numeric answer.\n\n"
        )
        
        # Add demonstrations if provided
        if demos:
            for demo in demos:
                prompt_parts.append(f"Question: {demo['question']}\n")
                if "reasoning" in demo:
                    prompt_parts.append(f"Answer: {demo['reasoning']}\n\n")
                else:
                    prompt_parts.append(f"Answer: The answer is {demo['answer']}.\n\n")
                    
        # Add target question
        prompt_parts.append(f"Question: {question}\n")
        prompt_parts.append("Answer:")
        
        return "".join(prompt_parts)


class ReliabilityScorer:
    """Compute reliability scores for CoT demonstrations."""
    
    def __init__(self, llm: LLMWrapper):
        self.llm = llm
        
    def compute_self_consistency(
        self,
        question: str,
        reference_answer: float,
        num_samples: int = 5,
        temperature: float = 0.7
    ) -> float:
        """Compute self-consistency score via sampling.
        
        r_sc = fraction of samples that agree with reference answer
        
        Args:
            question: Question to answer
            reference_answer: Reference answer to compare against
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            
        Returns:
            Self-consistency score in [0, 1]
        """
        if num_samples <= 1:
            return 1.0
            
        consistent_count = 0
        
        for _ in range(num_samples):
            generated = self.llm.generate_cot(question)
            predicted_answer = self._extract_answer(generated)
            
            # Check if answer matches reference
            if self._answers_match(predicted_answer, reference_answer):
                consistent_count += 1
                
        return consistent_count / num_samples
        
    def compute_paraphrase_invariance(
        self,
        question: str,
        reference_reasoning: str,
        reference_answer: float,
        num_paraphrases: int = 3
    ) -> float:
        """Compute paraphrase invariance score.
        
        r_pi = fraction of paraphrased questions that yield consistent answer
        
        Args:
            question: Original question
            reference_reasoning: Reference CoT reasoning
            reference_answer: Reference answer
            num_paraphrases: Number of paraphrases to generate
            
        Returns:
            Paraphrase invariance score in [0, 1]
        """
        if num_paraphrases <= 1:
            return 1.0
            
        consistent_count = 0
        
        for _ in range(num_paraphrases):
            # Generate paraphrase
            paraphrase = self._paraphrase_question(question)
            
            # Generate answer for paraphrased question
            generated = self.llm.generate_cot(paraphrase)
            predicted_answer = self._extract_answer(generated)
            
            # Check consistency
            if self._answers_match(predicted_answer, reference_answer):
                consistent_count += 1
                
        return consistent_count / num_paraphrases
        
    def compute_cycle_consistency(
        self,
        question: str,
        reasoning: str
    ) -> float:
        """Compute cycle consistency score.
        
        r_cc = can we reconstruct the original question from the reasoning?
        
        Strategy: Ask LLM to extract the question from reasoning, then check similarity.
        
        Args:
            question: Original question
            reasoning: CoT reasoning
            
        Returns:
            Cycle consistency score in [0, 1]
        """
        # Build prompt to reconstruct question from reasoning
        reconstruct_prompt = (
            f"Given the following reasoning, what was the original question?\n\n"
            f"Reasoning: {reasoning}\n\n"
            f"Original question:"
        )
        
        reconstructed = self.llm.generate(reconstruct_prompt, temperature=0.0)
        
        # Compute similarity (simple word overlap metric)
        similarity = self._compute_text_similarity(question, reconstructed)
        
        return similarity
        
    def compute_c3_reliability(
        self,
        question: str,
        reasoning: str,
        answer: float,
        sc_samples: int = 5,
        pi_paraphrases: int = 3,
        enable_cc: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """Compute C3 reliability score: r = r_sc * r_pi * r_cc.
        
        Args:
            question: Question text
            reasoning: CoT reasoning
            answer: Numeric answer
            sc_samples: Number of self-consistency samples
            pi_paraphrases: Number of paraphrases
            enable_cc: Whether to compute cycle consistency
            
        Returns:
            (overall_reliability, component_scores)
        """
        # Compute component scores
        r_sc = self.compute_self_consistency(question, answer, sc_samples)
        r_pi = self.compute_paraphrase_invariance(question, reasoning, answer, pi_paraphrases)
        r_cc = self.compute_cycle_consistency(question, reasoning) if enable_cc else 1.0
        
        # Overall reliability
        r_total = r_sc * r_pi * r_cc
        
        components = {
            "r_sc": r_sc,
            "r_pi": r_pi,
            "r_cc": r_cc,
            "r_total": r_total
        }
        
        return r_total, components
        
    def _extract_answer(self, text: str) -> float:
        """Extract numeric answer from generated text."""
        from preprocess import extract_answer_from_text
        return extract_answer_from_text(text)
        
    def _answers_match(self, pred: float, target: float, tolerance: float = 1e-3) -> bool:
        """Check if two answers match within tolerance."""
        if pred != pred or target != target:  # NaN check
            return False
        return abs(pred - target) < tolerance
        
    def _paraphrase_question(self, question: str) -> str:
        """Generate a paraphrase of the question."""
        prompt = (
            f"Paraphrase the following question while preserving its meaning:\n\n"
            f"Original: {question}\n\n"
            f"Paraphrase:"
        )
        paraphrase = self.llm.generate(prompt, temperature=0.7, max_new_tokens=128)
        return paraphrase.strip()
        
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple word-overlap similarity between two texts."""
        # Tokenize into words
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
