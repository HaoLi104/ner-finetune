#!/usr/bin/env python3
"""
Example usage of the FastAPI endpoints for entity evaluation.
"""

import requests
import json


def test_api_endpoints():
    """Test the FastAPI endpoints."""
    
    # API base URL
    base_url = "http://localhost:8000"
    
    print("=== FastAPI Entity Evaluation Examples ===\n")
    
    # Example 1: NER Evaluation
    print("1. NER Evaluation:")
    
    ner_request = {
        "y_true": [
            {
                "entities": [
                    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
                ]
            }
        ],
        "y_pred": [
            {
                "entities": [
                    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"},
                    {"start": 60, "end": 70, "type": "手术名称", "text": "额外手术"}
                ]
            }
        ],
        "mode": "ner",
        "strict_match": True,
        "enable_validation": False
    }
    
    try:
        response = requests.post(f"{base_url}/evaluate", json=ner_request)
        if response.status_code == 200:
            results = response.json()
            print(f"   Precision: {results['precision']:.3f}")
            print(f"   Recall: {results['recall']:.3f}")
            print(f"   F1 Score: {results['f1_score']:.3f}")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server. Make sure the server is running.")
    
    print()
    
    # Example 2: Semi-Structured Position Matching
    print("2. Semi-Structured Position Matching:")
    
    semi_position_request = {
        "y_true": [
            {
                "entities": [
                    {"start": 10, "end": 20, "text": "患者姓名"},
                    {"start": 30, "end": 50, "text": "45岁"}
                ]
            }
        ],
        "y_pred": [
            {
                "entities": [
                    {"start": 10, "end": 20, "text": "患者姓名"},
                    {"start": 35, "end": 55, "text": "45岁"},  # Position error
                    {"start": 60, "end": 70, "text": "床位号"}
                ]
            }
        ],
        "mode": "semi_structured",
        "matching_method": "position"
    }
    
    try:
        response = requests.post(f"{base_url}/evaluate", json=semi_position_request)
        if response.status_code == 200:
            results = response.json()
            print(f"   Precision: {results['precision']:.3f}")
            print(f"   Recall: {results['recall']:.3f}")
            print(f"   F1 Score: {results['f1_score']:.3f}")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server. Make sure the server is running.")
    
    print()
    
    # Example 3: Semi-Structured Text Exact Matching
    print("3. Semi-Structured Text Exact Matching:")
    
    semi_text_exact_request = {
        "y_true": [
            {
                "entities": [
                    {"start": 10, "end": 20, "text": "患者姓名"},
                    {"start": 30, "end": 50, "text": "45岁"}
                ]
            }
        ],
        "y_pred": [
            {
                "entities": [
                    {"start": 10, "end": 20, "text": "患者姓名"},
                    {"start": 35, "end": 55, "text": "45岁"},  # Same text, different position
                    {"start": 60, "end": 70, "text": "床位号"}
                ]
            }
        ],
        "mode": "semi_structured",
        "matching_method": "text",
        "text_match_mode": "exact"
    }
    
    try:
        response = requests.post(f"{base_url}/evaluate", json=semi_text_exact_request)
        if response.status_code == 200:
            results = response.json()
            print(f"   Precision: {results['precision']:.3f}")
            print(f"   Recall: {results['recall']:.3f}")
            print(f"   F1 Score: {results['f1_score']:.3f}")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server. Make sure the server is running.")
    
    print()
    
    # Example 4: Semi-Structured Text Overlap Matching
    print("4. Semi-Structured Text Overlap Matching:")
    
    semi_text_overlap_request = {
        "y_true": [
            {
                "entities": [
                    {"start": 10, "end": 20, "text": "患者姓名"},
                    {"start": 30, "end": 50, "text": "45岁"}
                ]
            }
        ],
        "y_pred": [
            {
                "entities": [
                    {"start": 10, "end": 20, "text": "患者姓名"},
                    {"start": 30, "end": 50, "text": "45岁"},
                    {"start": 60, "end": 70, "text": "患者名"}  # Overlaps with "患者姓名"
                ]
            }
        ],
        "mode": "semi_structured",
        "matching_method": "text",
        "text_match_mode": "overlap"
    }
    
    try:
        response = requests.post(f"{base_url}/evaluate", json=semi_text_overlap_request)
        if response.status_code == 200:
            results = response.json()
            print(f"   Precision: {results['precision']:.3f}")
            print(f"   Recall: {results['recall']:.3f}")
            print(f"   F1 Score: {results['f1_score']:.3f}")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server. Make sure the server is running.")
    
    print()
    
    # Example 5: Semi-Structured Text Similarity Matching
    print("5. Semi-Structured Text Similarity Matching:")
    
    semi_text_similarity_request = {
        "y_true": [
            {
                "entities": [
                    {"start": 10, "end": 20, "text": "患者姓名"},
                    {"start": 30, "end": 50, "text": "45岁"}
                ]
            }
        ],
        "y_pred": [
            {
                "entities": [
                    {"start": 10, "end": 20, "text": "患者姓名"},
                    {"start": 30, "end": 50, "text": "45岁"},
                    {"start": 60, "end": 70, "text": "患者名"}  # Similar to "患者姓名"
                ]
            }
        ],
        "mode": "semi_structured",
        "matching_method": "text",
        "text_match_mode": "similarity",
        "similarity_threshold": 0.8,
        "similarity_model": "tiny"
    }
    
    try:
        response = requests.post(f"{base_url}/evaluate", json=semi_text_similarity_request)
        if response.status_code == 200:
            results = response.json()
            print(f"   Precision: {results['precision']:.3f}")
            print(f"   Recall: {results['recall']:.3f}")
            print(f"   F1 Score: {results['f1_score']:.3f}")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server. Make sure the server is running.")
    
    print()
    
    # Example 6: Quick Evaluation
    print("6. Quick Evaluation:")
    
    quick_request = {
        "y_true": [
            {
                "entities": [
                    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
                ]
            }
        ],
        "y_pred": [
            {
                "entities": [
                    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
                ]
            }
        ],
        "mode": "ner"
    }
    
    try:
        response = requests.post(f"{base_url}/quick-evaluate", json=quick_request)
        if response.status_code == 200:
            results = response.json()
            print(f"   F1 Score: {results['f1_score']:.3f}")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server. Make sure the server is running.")
    
    print()
    
    # Example 7: Evaluation by Type
    print("7. Evaluation by Type:")
    
    type_request = {
        "y_true": [
            {
                "entities": [
                    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"},
                    {"start": 60, "end": 80, "type": "手术日期", "text": "2024-08-15"}
                ]
            }
        ],
        "y_pred": [
            {
                "entities": [
                    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
                ]
            }
        ],
        "mode": "ner",
        "strict_match": True,
        "enable_validation": False
    }
    
    try:
        response = requests.post(f"{base_url}/evaluate-by-type", json=type_request)
        if response.status_code == 200:
            results = response.json()
            print(f"   Overall F1: {results['overall']['f1_score']:.3f}")
            print("   By Type:")
            for entity_type, metrics in results['by_type'].items():
                print(f"     {entity_type}: F1={metrics['f1_score']:.3f}")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server. Make sure the server is running.")
    
    print()
    
    # Example 8: Health Check
    print("8. Health Check:")
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"   Status: {response.json()['status']}")
        else:
            print(f"   Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server. Make sure the server is running.")


def curl_examples():
    """Show curl examples for the API endpoints."""
    
    print("\n=== cURL Examples ===\n")
    
    print("1. NER Evaluation:")
    print("""curl -X POST "http://localhost:8000/evaluate" \\
     -H "Content-Type: application/json" \\
     -d '{
       "y_true": [{"entities": [{"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}]}],
       "y_pred": [{"entities": [{"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}]}],
       "mode": "ner"
     }'""")
    
    print("\n2. Semi-Structured Position Matching:")
    print("""curl -X POST "http://localhost:8000/evaluate" \\
     -H "Content-Type: application/json" \\
     -d '{
       "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
       "y_pred": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
       "mode": "semi_structured",
       "matching_method": "position"
     }'""")
    
    print("\n3. Semi-Structured Text Exact Matching:")
    print("""curl -X POST "http://localhost:8000/evaluate" \\
     -H "Content-Type: application/json" \\
     -d '{
       "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
       "y_pred": [{"entities": [{"start": 15, "end": 25, "text": "患者姓名"}]}],
       "mode": "semi_structured",
       "matching_method": "text",
       "text_match_mode": "exact"
     }'""")
    
    print("\n4. Semi-Structured Text Similarity Matching:")
    print("""curl -X POST "http://localhost:8000/evaluate" \\
     -H "Content-Type: application/json" \\
     -d '{
       "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
       "y_pred": [{"entities": [{"start": 15, "end": 25, "text": "患者名"}]}],
       "mode": "semi_structured",
       "matching_method": "text",
       "text_match_mode": "similarity",
       "similarity_threshold": 0.8,
       "similarity_model": "tiny"
     }'""")
    
    print("\n5. Quick Evaluation:")
    print("""curl -X POST "http://localhost:8000/quick-evaluate" \\
     -H "Content-Type: application/json" \\
     -d '{
       "y_true": [{"entities": [{"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}]}],
       "y_pred": [{"entities": [{"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}]}],
       "mode": "ner"
     }'""")
    
    print("\n6. Health Check:")
    print("curl -X GET \"http://localhost:8000/health\"")


if __name__ == "__main__":
    test_api_endpoints()
    curl_examples() 