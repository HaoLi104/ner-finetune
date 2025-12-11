# API Usage Guide

This guide explains how to use the FastAPI web service for entity evaluation.

## Overview

The API provides RESTful endpoints for evaluating NER and semi-structured entity extraction results. It supports all the same functionality as the Python library but through HTTP requests.

## Quick Start

### 1. Start the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python start_api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

### 2. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get API information
curl http://localhost:8000/
```

## API Endpoints

### 1. `/evaluate` - Full Evaluation

Evaluate entities with all available options.

**Method**: `POST`

**Request Body**:
```json
{
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
  "mode": "ner",
  "strict_match": true,
  "enable_validation": false
}
```

**Response**:
```json
{
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0,
  "true_positives": 2,
  "false_positives": 0,
  "false_negatives": 0,
  "accuracy": 1.0,
  "true_negatives": 0
}
```

### 2. `/quick-evaluate` - Quick Evaluation

Quick evaluation with default settings.

**Method**: `POST`

**Request Body**:
```json
{
  "y_true": [
    {
      "entities": [
        {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}
      ]
    }
  ],
  "y_pred": [
    {
      "entities": [
        {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}
      ]
    }
  ],
  "mode": "ner"
}
```

### 3. `/evaluate-by-type` - Type-Specific Evaluation

Evaluate NER entities by type (NER mode only).

**Method**: `POST`

**Request Body**:
```json
{
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
```

**Response**:
```json
{
  "overall": {
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "true_positives": 2,
    "false_positives": 0,
    "false_negatives": 0,
    "accuracy": 1.0,
    "true_negatives": 0
  },
  "by_type": {
    "手术日期": {
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0
    },
    "手术名称": {
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0
    }
  }
}
```

### 4. `/health` - Health Check

Check if the API is running.

**Method**: `GET`

**Response**:
```json
{
  "status": "healthy"
}
```

## Semi-Structured Evaluation Examples

### Position Matching

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
    "y_pred": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
    "mode": "semi_structured",
    "matching_method": "position"
  }'
```

### Text Exact Matching

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
    "y_pred": [{"entities": [{"start": 15, "end": 25, "text": "患者姓名"}]}],
    "mode": "semi_structured",
    "matching_method": "text",
    "text_match_mode": "exact"
  }'
```

### Text Overlap Matching

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
    "y_pred": [{"entities": [{"start": 10, "end": 18, "text": "患者名"}]}],
    "mode": "semi_structured",
    "matching_method": "text",
    "text_match_mode": "overlap"
  }'
```

### Text Similarity Matching

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
    "y_pred": [{"entities": [{"start": 15, "end": 25, "text": "患者名"}]}],
    "mode": "semi_structured",
    "matching_method": "text",
    "text_match_mode": "similarity",
    "similarity_threshold": 0.8,
    "similarity_model": "tiny"
  }'
```

## Python Client Examples

### Using requests

```python
import requests

# NER evaluation
response = requests.post("http://localhost:8000/evaluate", json={
    "y_true": [{"entities": [{"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}]}],
    "y_pred": [{"entities": [{"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}]}],
    "mode": "ner"
})

results = response.json()
print(f"F1 Score: {results['f1_score']:.3f}")
```

### Using the examples

```bash
# Run the example script
python examples/api_usage.py
```

## Error Handling

### Common Error Responses

**400 Bad Request**:
```json
{
  "detail": "Invalid request parameters"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Internal server error: Error message"
}
```

### Error Handling in Python

```python
import requests

try:
    response = requests.post("http://localhost:8000/evaluate", json=request_data)
    response.raise_for_status()  # Raises an exception for 4XX/5XX status codes
    results = response.json()
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except ValueError as e:
    print(f"Invalid JSON response: {e}")
```

## Data Format

### Entity Format

```json
{
  "start": 10,        // Start position (required)
  "end": 20,          // End position (required)
  "type": "手术日期",   // Entity type (NER mode only)
  "text": "2024-09-10" // Entity text (required)
}
```

### Document Format

```json
{
  "entities": [
    // Array of entities
  ]
}
```

### Request Format

```json
{
  "y_true": [
    // Array of documents with ground truth entities
  ],
  "y_pred": [
    // Array of documents with predicted entities
  ],
  "mode": "ner",                    // "ner" or "semi_structured"
  "strict_match": true,             // NER mode: strict or overlap matching
  "enable_validation": false,       // Enable entity validation
  "matching_method": "position",    // Semi-structured: "position" or "text"
  "text_match_mode": "exact",       // Semi-structured text: "exact", "overlap", "similarity"
  "similarity_threshold": 0.8,      // Similarity threshold (0.0-1.0)
  "similarity_model": "tiny"        // Model: "tiny", "small", "medium", "large", "offline"
}
```

## Performance Considerations

### Batch Processing

For large datasets, consider processing in batches:

```python
def evaluate_batch(documents, batch_size=100):
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        response = requests.post("http://localhost:8000/evaluate", json=batch)
        results.append(response.json())
    return results
```

### Model Loading

When using similarity matching, the first request may take longer due to model loading. Subsequent requests will be faster.

## Security Considerations

### Production Deployment

For production use:

1. **Use HTTPS**: Configure SSL/TLS certificates
2. **Authentication**: Add API key or token authentication
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Input Validation**: Validate input data size and format
5. **Error Handling**: Don't expose internal error details

### Example with Authentication

```python
import requests

headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://your-api-domain.com/evaluate",
    json=request_data,
    headers=headers
)
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure the API server is running
2. **Model Download Issues**: Check internet connection for similarity models
3. **Memory Issues**: Use smaller models or offline mode
4. **Timeout**: Increase timeout for large datasets

### Debug Mode

Start the server in debug mode:

```bash
python start_api.py
```

Check the server logs for detailed error information.

## API Documentation

For interactive API documentation, visit:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

These provide:
- Interactive testing interface
- Request/response schemas
- Example requests
- Error codes and descriptions 