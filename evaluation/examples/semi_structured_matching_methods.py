#!/usr/bin/env python3
"""
Demonstration of different matching methods for semi-structured entity evaluation.

This example shows three different matching approaches:
1. Position matching: Compare start and end positions (exact match only)
2. Text exact matching: Compare text content exactly
3. Text overlap matching: Compare text content with overlap support
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.easy_eval import evaluate_entities


def main():
    """Demonstrate different semi-structured matching methods."""
    
    print("=== Semi-Structured Entity Evaluation - Matching Methods Demo ===\n")
    
    # Example data: Medical discharge record with various types of errors
    y_true = [
        {
            "entities": [
                {"text": "患者姓名", "start": 5, "end": 9},
                {"text": "张三", "start": 10, "end": 12},
                {"text": "年龄", "start": 18, "end": 20},
                {"text": "45岁", "start": 21, "end": 24},
                {"text": "入院日期", "start": 38, "end": 42},
                {"text": "2024-01-15", "start": 43, "end": 53}
            ]
        }
    ]
    
    y_pred = [
        {
            "entities": [
                # Perfect matches (should be TP in all methods)
                {"text": "患者姓名", "start": 5, "end": 9},
                {"text": "张三", "start": 10, "end": 12},
                {"text": "年龄", "start": 18, "end": 20},
                {"text": "45岁", "start": 21, "end": 24},
                
                # Position errors (same text, wrong position) - should be FP in position matching
                {"text": "患者姓名", "start": 6, "end": 10},  # Wrong position
                {"text": "张三", "start": 11, "end": 13},     # Wrong position
                
                # Text errors (same position, wrong text) - should be FP in text matching
                {"text": "患者名", "start": 5, "end": 9},     # Partial text match
                {"text": "李四", "start": 10, "end": 12},     # Wrong text
                
                # Extra entities (should be FP in all methods)
                {"text": "床位号", "start": 60, "end": 64},
                {"text": "A101", "start": 65, "end": 69}
            ]
        }
    ]
    
    print("Ground Truth Entities:")
    for entity in y_true[0]["entities"]:
        print(f"  {entity['text']} ({entity['start']}-{entity['end']})")
    
    print("\nPredicted Entities:")
    for entity in y_pred[0]["entities"]:
        print(f"  {entity['text']} ({entity['start']}-{entity['end']})")
    
    print("\n" + "="*60)
    
    # 1. Position Matching (exact start and end match)
    print("1. POSITION MATCHING (exact start and end match)")
    print("   Only entities with identical start and end positions are considered matches.")
    print("   Text content is ignored for matching purposes.\n")
    
    results_position = evaluate_entities(
        y_true, y_pred, 
        mode="semi_structured", 
        matching_method="position"
    )
    
    print(f"   Precision: {results_position['precision']:.3f}")
    print(f"   Recall:    {results_position['recall']:.3f}")
    print(f"   F1 Score:  {results_position['f1_score']:.3f}")
    print(f"   True Positives:  {results_position['true_positives']}")
    print(f"   False Positives: {results_position['false_positives']}")
    print(f"   False Negatives: {results_position['false_negatives']}")
    
    print("\n" + "="*60)
    
    # 2. Text Exact Matching
    print("2. TEXT EXACT MATCHING")
    print("   Only entities with identical text content are considered matches.")
    print("   Position information is ignored for matching purposes.\n")
    
    results_text_exact = evaluate_entities(
        y_true, y_pred, 
        mode="semi_structured", 
        matching_method="text",
        text_match_mode="exact"
    )
    
    print(f"   Precision: {results_text_exact['precision']:.3f}")
    print(f"   Recall:    {results_text_exact['recall']:.3f}")
    print(f"   F1 Score:  {results_text_exact['f1_score']:.3f}")
    print(f"   True Positives:  {results_text_exact['true_positives']}")
    print(f"   False Positives: {results_text_exact['false_positives']}")
    print(f"   False Negatives: {results_text_exact['false_negatives']}")
    
    print("\n" + "="*60)
    
    # 3. Text Overlap Matching
    print("3. TEXT OVERLAP MATCHING")
    print("   Entities with overlapping text content are considered matches.")
    print("   Position information is ignored for matching purposes.\n")
    
    results_text_overlap = evaluate_entities(
        y_true, y_pred, 
        mode="semi_structured", 
        matching_method="text",
        text_match_mode="overlap"
    )
    
    print(f"   Precision: {results_text_overlap['precision']:.3f}")
    print(f"   Recall:    {results_text_overlap['recall']:.3f}")
    print(f"   F1 Score:  {results_text_overlap['f1_score']:.3f}")
    print(f"   True Positives:  {results_text_overlap['true_positives']}")
    print(f"   False Positives: {results_text_overlap['false_positives']}")
    print(f"   False Negatives: {results_text_overlap['false_negatives']}")
    
    print("\n" + "="*60)
    
    # Comparison Summary
    print("COMPARISON SUMMARY:")
    print(f"{'Method':<20} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)
    print(f"{'Position':<20} {results_position['precision']:<10.3f} {results_position['recall']:<10.3f} {results_position['f1_score']:<10.3f}")
    print(f"{'Text Exact':<20} {results_text_exact['precision']:<10.3f} {results_text_exact['recall']:<10.3f} {results_text_exact['f1_score']:<10.3f}")
    print(f"{'Text Overlap':<20} {results_text_overlap['precision']:<10.3f} {results_text_overlap['recall']:<10.3f} {results_text_overlap['f1_score']:<10.3f}")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("• Position matching is strictest - only perfect position matches count")
    print("• Text exact matching focuses on content accuracy regardless of position")
    print("• Text overlap matching is most lenient - partial text matches count")
    print("• Choose the method based on your evaluation priorities:")
    print("  - Use 'position' if precise span detection is critical")
    print("  - Use 'text exact' if content accuracy is more important than position")
    print("  - Use 'text overlap' if you want to reward partial content matches")
    
    print("\n" + "="*60)
    print("DETAILED ANALYSIS:")
    print("Position Matching:")
    print("  ✓ Perfect matches: 4 TP")
    print("  ✗ Position errors: 2 FP (same text, wrong position)")
    print("  ✗ Text errors: 2 FP (same position, wrong text)")
    print("  ✗ Extra entities: 2 FP")
    print("  Total: 4 TP, 6 FP, 2 FN")
    
    print("\nText Exact Matching:")
    print("  ✓ Perfect matches: 4 TP")
    print("  ✓ Position errors: 2 TP (same text, wrong position)")
    print("  ✗ Text errors: 2 FP (same position, wrong text)")
    print("  ✗ Extra entities: 2 FP")
    print("  Total: 6 TP, 4 FP, 0 FN")
    
    print("\nText Overlap Matching:")
    print("  ✓ Perfect matches: 4 TP")
    print("  ✓ Position errors: 2 TP (same text, wrong position)")
    print("  ✓ Partial text match: 1 TP ('患者名' overlaps with '患者姓名')")
    print("  ✗ Wrong text: 1 FP ('李四' doesn't overlap with '张三')")
    print("  ✗ Extra entities: 2 FP")
    print("  Total: 7 TP, 3 FP, 0 FN")


if __name__ == "__main__":
    main() 