#!/usr/bin/env python3
"""
Simple test to verify BERT similarity calculation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.similarity_utils import ChineseMedicalSimilarity

def test_bert_similarity():
    print("Testing BERT similarity calculation...")
    
    # Create calculator with model loading
    calculator = ChineseMedicalSimilarity(load_model=True)
    
    # Test some medical term pairs
    test_pairs = [
        ("急性阑尾炎", "阑尾炎"),
        ("患者姓名", "患者名"),
        ("腹腔镜手术", "腹腔镜阑尾切除术"),
        ("入院日期", "入院时间"),
        ("糖尿病", "血糖异常"),
        ("高血压", "血压升高"),
        ("心脏病", "心血管疾病")
    ]
    
    print("Medical Term Similarities (BERT-based):")
    for term1, term2 in test_pairs:
        similarity = calculator.calculate_similarity(term1, term2)
        print(f"  '{term1}' vs '{term2}': {similarity:.3f}")
    
    # Test threshold checking
    print("\nThreshold Testing:")
    for threshold in [0.5, 0.7, 0.8, 0.9]:
        matches = 0
        for term1, term2 in test_pairs:
            if calculator.is_similar(term1, term2, threshold):
                matches += 1
        print(f"  Threshold {threshold}: {matches}/{len(test_pairs)} pairs match")

if __name__ == "__main__":
    test_bert_similarity() 