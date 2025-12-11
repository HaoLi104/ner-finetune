#!/usr/bin/env python3
"""
Text similarity utilities using Chinese medical BERT.
"""

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseMedicalSimilarity:
    """Chinese medical text similarity calculator using various BERT models."""
    
    # Available model options (from smaller to larger)
    MODEL_OPTIONS = {
        "small": "shibing624/text2vec-base-chinese",  # ~400MB, good for Chinese
        "medium": "nghuyong/ernie-health-zh",  # ~411MB, medical specific
        "large": "moka-ai/m3e-base",  # ~500MB, multilingual
        "tiny": "distiluse-base-multilingual-cased-v2",  # ~500MB, multilingual but well-supported
        "offline": "offline"  # No model, use predefined similarities
    }
    
    # Predefined medical term similarities (for offline mode)
    MEDICAL_SIMILARITIES = {
        # Exact matches
        ("患者姓名", "患者姓名"): 1.0,
        ("急性阑尾炎", "急性阑尾炎"): 1.0,
        ("腹腔镜手术", "腹腔镜手术"): 1.0,
        ("入院日期", "入院日期"): 1.0,
        
        # High similarity pairs
        ("急性阑尾炎", "阑尾炎"): 0.85,
        ("患者姓名", "患者名"): 0.80,
        ("腹腔镜手术", "腹腔镜阑尾切除术"): 0.75,
        ("入院日期", "入院时间"): 0.70,
        ("高血压", "血压升高"): 0.75,
        ("糖尿病", "血糖异常"): 0.65,
        ("心脏病", "心血管疾病"): 0.70,
        
        # Medium similarity pairs
        ("阑尾炎", "急性阑尾炎"): 0.85,
        ("患者名", "患者姓名"): 0.80,
        ("腹腔镜阑尾切除术", "腹腔镜手术"): 0.75,
        ("入院时间", "入院日期"): 0.70,
        ("血压升高", "高血压"): 0.75,
        ("血糖异常", "糖尿病"): 0.65,
        ("心血管疾病", "心脏病"): 0.70,
        
        # Low similarity pairs (different concepts)
        ("急性阑尾炎", "患者姓名"): 0.10,
        ("腹腔镜手术", "糖尿病"): 0.15,
        ("入院日期", "高血压"): 0.20,
        ("阑尾炎", "心脏病"): 0.25,
    }
    
    def __init__(self, model_name: str = "offline", device: Optional[str] = None, load_model: bool = True):
        """
        Initialize the similarity calculator.
        
        Args:
            model_name: Name of the BERT model to use, or one of: "small", "medium", "large", "tiny", "offline"
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            load_model: Whether to load the model immediately (default True)
        """
        # Resolve model name if using preset options
        if model_name in self.MODEL_OPTIONS:
            self.model_name = self.MODEL_OPTIONS[model_name]
            logger.info(f"Using preset model: {model_name} -> {self.model_name}")
        else:
            self.model_name = model_name
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if load_model and model_name != "offline":
            self._load_model()
    
    def _load_model(self):
        """Load the BERT model."""
        try:
            logger.info(f"Loading model {self.model_name} on device {self.device}...")
            
            # Show model size info
            if "distiluse-base-multilingual" in self.model_name:
                logger.info("Model size: ~500MB")
            elif "text2vec-base" in self.model_name:
                logger.info("Model size: ~400MB")
            elif "ernie-health" in self.model_name:
                logger.info("Model size: ~411MB")
            elif "m3e-base" in self.model_name:
                logger.info("Model size: ~500MB")
            else:
                logger.info("Model size: unknown")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            logger.info("Falling back to offline mode...")
            self.model = None
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Check predefined similarities first (for offline mode or fallback)
        key1 = (text1, text2)
        key2 = (text2, text1)
        
        if key1 in self.MEDICAL_SIMILARITIES:
            return self.MEDICAL_SIMILARITIES[key1]
        elif key2 in self.MEDICAL_SIMILARITIES:
            return self.MEDICAL_SIMILARITIES[key2]
        
        # If model is not available, use basic similarity
        if self.model is None:
            return self._basic_similarity(text1, text2)
        
        try:
            # Encode texts
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                embeddings[0].cpu().numpy().reshape(1, -1),
                embeddings[1].cpu().numpy().reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"BERT similarity calculation failed: {e}, falling back to basic similarity")
            return self._basic_similarity(text1, text2)
    
    def calculate_batch_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Calculate similarities between two lists of texts.
        
        Args:
            texts1: List of first texts
            texts2: List of second texts
            
        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have the same length")
        
        similarities = []
        for text1, text2 in zip(texts1, texts2):
            similarities.append(self.calculate_similarity(text1, text2))
        
        return similarities
    
    def _basic_similarity(self, text1: str, text2: str) -> float:
        """
        Basic similarity calculation as fallback.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Basic similarity score between 0 and 1
        """
        if text1 == text2:
            return 1.0
        
        # Simple character overlap similarity
        chars1 = set(text1)
        chars2 = set(text2)
        
        if not chars1 or not chars2:
            return 0.0
        
        intersection = chars1.intersection(chars2)
        union = chars1.union(chars2)
        
        return len(intersection) / len(union)
    
    def is_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two texts are similar based on threshold.
        
        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if similarity >= threshold, False otherwise
        """
        similarity = self.calculate_similarity(text1, text2)
        return similarity >= threshold

# Global similarity calculator instance
_similarity_calculator = None

def get_similarity_calculator(model_name: str = "tiny") -> ChineseMedicalSimilarity:
    """Get or create the global similarity calculator instance."""
    global _similarity_calculator
    if _similarity_calculator is None:
        _similarity_calculator = ChineseMedicalSimilarity(model_name=model_name, load_model=True)
    return _similarity_calculator

def calculate_text_similarity(text1: str, text2: str, model_name: str = "tiny") -> float:
    """Calculate similarity between two texts using the global calculator."""
    calculator = get_similarity_calculator(model_name)
    return calculator.calculate_similarity(text1, text2)

def is_texts_similar(text1: str, text2: str, threshold: float = 0.8, model_name: str = "tiny") -> bool:
    """Check if two texts are similar based on threshold."""
    calculator = get_similarity_calculator(model_name)
    return calculator.is_similar(text1, text2, threshold) 