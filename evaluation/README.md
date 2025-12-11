# Entity Evaluation Project

This is a Python project for both Named Entity Recognition (NER) and Semi-Structured entity evaluation with comprehensive validation features.

## Project Structure

```
evaluation/
├── src/                    # Source code
│   ├── ner_metrics.py     # NER and semi-structured evaluation metrics
│   ├── easy_eval.py       # Easy-to-use evaluation functions
│   ├── similarity_utils.py # BERT similarity calculation utilities
│   └── main.py           # Main entry point with demos
├── tests/                 # Test files
├── examples/              # Usage examples
│   ├── semi_structured_matching_methods.py # Three matching methods demo
│   ├── similarity_matching_demo.py # BERT similarity demo
│   └── api_usage.py       # FastAPI usage examples
├── docs/                  # Documentation
│   └── quick_start.md     # Quick start guide
├── requirements.txt
├── setup.py
└── README.md
```

## Features

- **NER Evaluation Metrics**: Precision, Recall, F1 Score for medical NER
- **Semi-Structured Evaluation**: Precision, Reall, F1 Score for semi-Structred entity extraction evaluation with 4 matching methods
- **BERT Text Similarity**: Semantic similarity matching using pre-trained BERT models
- **Configurable Entity Validation**: Optional validation to ensure start, end, text fields match between true and predicted entities
- **Flexible Matching**: Supports multiple matching strategies:
  - NER: strict (exact) and overlap matching
  - Semi-structured: position matching, text exact matching, text overlap matching, and text similarity matching
- **Type-specific Metrics**: Calculate metrics for each entity type separately (NER mode)
- **Comprehensive Testing**: Extensive test coverage for validation scenarios
- **FastAPI Web Service**: RESTful API for easy integration with web applications
- **Simple Python API**: Easy-to-use functions for direct Python integration

## Installation

```bash
pip install -r requirements.txt
```

## BERT Model Setup

This project uses BERT models for text similarity matching. The models will be automatically downloaded on first use.

### Model Options

The following model presets are available:

- **`"tiny"`** (default): `distiluse-base-multilingual-cased-v2` (~500MB)
  - Fast, multilingual, good for most use cases
- **`"small"`**: `shibing624/text2vec-small-chinese` (~100MB)
  - Optimized for Chinese text
- **`"medium"`**: `shibing624/text2vec-base-chinese` (~400MB)
  - Better performance for Chinese text
- **`"large"`**: `nghuyong/ernie-health-zh` (~1.2GB)
  - Medical domain optimized, best performance
- **`"offline"`**: Uses predefined similarity scores
  - No model download required, works offline

### Model Download Location

Models are automatically downloaded to:
- **HuggingFace Hub Cache**: `~/.cache/huggingface/hub/`
- **SentenceTransformers Cache**: `~/.cache/torch/sentence_transformers/`

### Usage Examples

```python
from src.easy_eval import evaluate_entities

# Use default tiny model
results = evaluate_entities(
    y_true, y_pred,
    mode="semi_structured",
    matching_method="text",
    text_match_mode="similarity",
    similarity_threshold=0.8,
    similarity_model="tiny"  # Default, ~500MB
)

# Use Chinese-optimized model
results = evaluate_entities(
    y_true, y_pred,
    mode="semi_structured",
    matching_method="text",
    text_match_mode="similarity",
    similarity_threshold=0.8,
    similarity_model="medium"  # ~400MB, better for Chinese
)

# Use medical domain model
results = evaluate_entities(
    y_true, y_pred,
    mode="semi_structured",
    matching_method="text",
    text_match_mode="similarity",
    similarity_threshold=0.8,
    similarity_model="large"  # ~1.2GB, medical optimized
)

# Offline mode (no model download)
results = evaluate_entities(
    y_true, y_pred,
    mode="semi_structured",
    matching_method="text",
    text_match_mode="similarity",
    similarity_threshold=0.8,
    similarity_model="offline"  # No download required
)
```

### Model Management

```python
from src.similarity_utils import ChineseMedicalSimilarity

# Check available models
print(ChineseMedicalSimilarity.MODEL_OPTIONS)

# Load model manually
calculator = ChineseMedicalSimilarity(model_name="tiny")
similarity = calculator.calculate_similarity("患者姓名", "患者名")
print(f"Similarity: {similarity:.3f}")
```

### Troubleshooting

1. **Model Download Issues**:
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/hub/
   rm -rf ~/.cache/torch/sentence_transformers/
   ```

2. **Memory Issues**: Use smaller models (`"tiny"` or `"small"`)

3. **Offline Usage**: Use `similarity_model="offline"` for environments without internet access

4. **Custom Model Path**: Set environment variable:
   ```bash
   export TRANSFORMERS_CACHE="/path/to/custom/cache"
   ```

## Usage

### Simple Usage (Recommended)

```python
from src.easy_eval import evaluate_entities, quick_eval

# Prepare your data
y_true = [{"entities": [
    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
]}]

y_pred = [{"entities": [
    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"},
    {"start": 60, "end": 70, "type": "手术名称", "text": "额外手术"}  # False positive
]}]

# Method 1: Main function with all options
results = evaluate_entities(y_true, y_pred, mode="ner", strict_match=True, enable_validation=False)
print(f"F1 Score: {results['f1_score']:.3f}")

# Method 2: Quick evaluation with defaults
results = quick_eval(y_true, y_pred, mode="ner")
print(f"F1 Score: {results['f1_score']:.3f}")

# Method 3: Semi-structured evaluation with different matching methods
y_true_semi = [{"entities": [
    {"start": 10, "end": 20, "text": "患者姓名"},
    {"start": 30, "end": 50, "text": "45岁"}
]}]

y_pred_semi = [{"entities": [
    {"start": 10, "end": 20, "text": "患者姓名"},
    {"start": 35, "end": 55, "text": "45岁"},  # Position error
    {"start": 60, "end": 70, "text": "床位号"}  # Extra entity
]}]

# Position matching (exact start/end positions)
results = evaluate_entities(y_true_semi, y_pred_semi, mode="semi_structured", matching_method="position")
print(f"Position matching F1: {results['f1_score']:.3f}")

# Text exact matching (same text content, ignore position)
results = evaluate_entities(y_true_semi, y_pred_semi, mode="semi_structured", matching_method="text", text_match_mode="exact")
print(f"Text exact matching F1: {results['f1_score']:.3f}")

# Text overlap matching (overlapping text content)
results = evaluate_entities(y_true_semi, y_pred_semi, mode="semi_structured", matching_method="text", text_match_mode="overlap")
print(f"Text overlap matching F1: {results['f1_score']:.3f}")

# Text similarity matching (BERT-based semantic similarity)
results = evaluate_entities(
    y_true_semi, y_pred_semi, 
    mode="semi_structured", 
    matching_method="text", 
    text_match_mode="similarity",
    similarity_threshold=0.8,
    similarity_model="tiny"
)
print(f"Text similarity matching F1: {results['f1_score']:.3f}")



### Advanced Usage

```python
from src.ner_metrics import NERMetrics, SemiStructuredMetrics

# NER evaluation with custom settings
metrics = NERMetrics(strict_match=True, enable_validation=True)
results = metrics.calculate_metrics(y_true, y_pred)

# Semi-structured evaluation with different matching methods
# Position matching (default)
metrics = SemiStructuredMetrics(matching_method="position", enable_validation=False)
results = metrics.calculate_metrics(y_true_semi, y_pred_semi)

# Text exact matching
metrics = SemiStructuredMetrics(matching_method="text", text_match_mode="exact", enable_validation=False)
results = metrics.calculate_metrics(y_true_semi, y_pred_semi)

# Text overlap matching
metrics = SemiStructuredMetrics(matching_method="text", text_match_mode="overlap", enable_validation=False)
results = metrics.calculate_metrics(y_true_semi, y_pred_semi)

# Text similarity matching
metrics = SemiStructuredMetrics(
    matching_method="text", 
    text_match_mode="similarity", 
    similarity_threshold=0.8,
    similarity_model="tiny",
    enable_validation=False
)
results = metrics.calculate_metrics(y_true_semi, y_pred_semi)



### Entity Validation

The system provides optional validation that can be enabled to check:
- Entities with same start/end positions have matching types
- No conflicting boundaries (same start but different end, or vice versa)
- Allows false positives and false negatives (normal NER scenarios)

Validation is disabled by default. Enable it with `enable_validation=True`:

```python
# With validation enabled
metrics = NERMetrics(enable_validation=True)

# Without validation (default)
metrics = NERMetrics(enable_validation=False)

```

See [Validation Guide](docs/validation_guide.md) for detailed information.

## Semi-Structured Matching Methods

The semi-structured evaluation supports four different matching strategies:

### 1. Position Matching (`matching_method="position"`)
- **Purpose**: Focus on precise span detection
- **Logic**: Only entities with identical start and end positions are considered matches
- **Use case**: When exact position accuracy is critical
- **Example**: 
  - True: `{"text": "张三", "start": 10, "end": 12}`
  - Pred: `{"text": "张三", "start": 10, "end": 12}` ✓ (TP)
  - Pred: `{"text": "张三", "start": 11, "end": 13}` ✗ (FP)

### 2. Text Exact Matching (`matching_method="text", text_match_mode="exact"`)
- **Purpose**: Focus on content accuracy regardless of position
- **Logic**: Only entities with identical text content are considered matches
- **Use case**: When content accuracy is more important than position
- **Example**:
  - True: `{"text": "张三", "start": 10, "end": 12}`
  - Pred: `{"text": "张三", "start": 10, "end": 12}` ✓ (TP)
  - Pred: `{"text": "张三", "start": 15, "end": 17}` ✓ (TP - same text, different position)
  - Pred: `{"text": "李四", "start": 10, "end": 12}` ✗ (FP - same position, different text)

### 3. Text Overlap Matching (`matching_method="text", text_match_mode="overlap"`)
- **Purpose**: Allow partial text matches
- **Logic**: Entities with overlapping text content are considered matches
- **Use case**: When you want to reward partial content matches
- **Example**:
  - True: `{"text": "患者姓名", "start": 5, "end": 9}`
  - Pred: `{"text": "患者姓名", "start": 5, "end": 9}` ✓ (TP)
  - Pred: `{"text": "患者名", "start": 5, "end": 8}` ✓ (TP - "患者名" overlaps with "患者姓名")
  - Pred: `{"text": "姓名", "start": 7, "end": 9}` ✓ (TP - "姓名" overlaps with "患者姓名")

### 4. Text Similarity Matching (`matching_method="text", text_match_mode="similarity"`)
- **Purpose**: Semantic similarity matching using BERT embeddings
- **Logic**: Entities with similar semantic meaning are considered matches
- **Use case**: When you want to match semantically similar but textually different entities
- **Example**:
  - True: `{"text": "患者姓名", "start": 5, "end": 9}`
  - Pred: `{"text": "患者名", "start": 5, "end": 8}` ✓ (TP - semantically similar)
  - Pred: `{"text": "姓名", "start": 7, "end": 9}` ✓ (TP - semantically similar)
  - Pred: `{"text": "手术日期", "start": 5, "end": 9}` ✗ (FP - semantically different)

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run demo
python src/main.py

# Run specific examples
python examples/semi_structured_matching_methods.py
python examples/similarity_matching_demo.py
python examples/api_usage.py

# Start FastAPI server (if available)
python start_api.py
```

## FastAPI Web Service

### Start the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python start_api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

### API Endpoints

- `POST /evaluate` - Evaluate entities with full options
- `POST /quick-evaluate` - Quick evaluation with defaults
- `POST /evaluate-by-type` - Evaluate NER entities by type
- `GET /health` - Health check
- `GET /` - API information

### Example API Usage

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

# Semi-structured evaluation with position matching
response = requests.post("http://localhost:8000/evaluate", json={
    "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
    "y_pred": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
    "mode": "semi_structured",
    "matching_method": "position"
})

results = response.json()
print(f"Position matching F1: {results['f1_score']:.3f}")

# Semi-structured evaluation with text exact matching
response = requests.post("http://localhost:8000/evaluate", json={
    "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
    "y_pred": [{"entities": [{"start": 15, "end": 25, "text": "患者姓名"}]}],
    "mode": "semi_structured",
    "matching_method": "text",
    "text_match_mode": "exact"
})

results = response.json()
print(f"Text exact matching F1: {results['f1_score']:.3f}")

# Semi-structured evaluation with text similarity matching
response = requests.post("http://localhost:8000/evaluate", json={
    "y_true": [{"entities": [{"start": 10, "end": 20, "text": "患者姓名"}]}],
    "y_pred": [{"entities": [{"start": 15, "end": 25, "text": "患者名"}]}],
    "mode": "semi_structured",
    "matching_method": "text",
    "text_match_mode": "similarity",
    "similarity_threshold": 0.8,
    "similarity_model": "tiny"
})

results = response.json()
print(f"Text similarity matching F1: {results['f1_score']:.3f}")
```

See [examples/api_usage.py](examples/api_usage.py) for more examples.

## Documentation

- [Quick Start Guide](docs/quick_start.md) - Get started quickly with all evaluation modes
- [Model Setup Guide](docs/model_setup_guide.md) - Complete guide for BERT model setup and usage
- [API Usage Guide](docs/api_usage_guide.md) - Complete guide for using the FastAPI web service
- [Examples](examples/) - Complete usage examples for all features
- [API Documentation](docs/) - Complete API reference 