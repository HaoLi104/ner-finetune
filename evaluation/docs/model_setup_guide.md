# BERT Model Setup Guide

This guide explains how to set up and use BERT models for text similarity matching in the evaluation project.

## Overview

The project uses pre-trained BERT models to calculate semantic similarity between text entities. Models are automatically downloaded on first use and cached locally.

## Model Options

### Available Models

| Model Name | Size | Description | Use Case |
|------------|------|-------------|----------|
| `"tiny"` | ~500MB | `distiluse-base-multilingual-cased-v2` | Default, multilingual, fast |
| `"small"` | ~100MB | `shibing624/text2vec-small-chinese` | Chinese optimized, small |
| `"medium"` | ~400MB | `shibing624/text2vec-base-chinese` | Chinese optimized, better performance |
| `"large"` | ~1.2GB | `nghuyong/ernie-health-zh` | Medical domain, best performance |
| `"offline"` | 0MB | Predefined similarities | No download, offline use |

### Model Selection Guide

- **For general use**: Use `"tiny"` (default)
- **For Chinese text**: Use `"small"` or `"medium"`
- **For medical domain**: Use `"large"`
- **For offline environments**: Use `"offline"`

## Installation and Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. First Use (Automatic Download)

Models are downloaded automatically on first use:

```python
from src.easy_eval import evaluate_entities

# This will trigger model download
results = evaluate_entities(
    y_true, y_pred,
    mode="semi_structured",
    matching_method="text",
    text_match_mode="similarity",
    similarity_model="tiny"  # Will download ~500MB
)
```

### 3. Verify Installation

```python
from src.similarity_utils import ChineseMedicalSimilarity

# Test model loading
calculator = ChineseMedicalSimilarity(model_name="tiny")
similarity = calculator.calculate_similarity("患者姓名", "患者名")
print(f"Similarity: {similarity:.3f}")
```

## Model Storage Locations

### Default Cache Directories

Models are stored in the following locations:

- **HuggingFace Hub Cache**: `~/.cache/huggingface/hub/`
- **SentenceTransformers Cache**: `~/.cache/torch/sentence_transformers/`

### Check Model Location

```bash
# Find downloaded models
find ~/.cache -name "*distiluse*" -type d
find ~/.cache -name "*text2vec*" -type d
find ~/.cache -name "*ernie*" -type d
```

### Custom Cache Location

Set environment variables to change cache location:

```bash
# Set custom cache directory
export TRANSFORMERS_CACHE="/path/to/custom/cache"
export HF_HOME="/path/to/custom/huggingface"

# Or in Python
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/custom/cache'
```

## Usage Examples

### Basic Usage

```python
from src.easy_eval import evaluate_entities

# Use default tiny model
results = evaluate_entities(
    y_true, y_pred,
    mode="semi_structured",
    matching_method="text",
    text_match_mode="similarity",
    similarity_threshold=0.8,
    similarity_model="tiny"
)
```

### Advanced Usage

```python
from src.similarity_utils import ChineseMedicalSimilarity

# Load model manually
calculator = ChineseMedicalSimilarity(model_name="medium")

# Calculate similarity between texts
similarity = calculator.calculate_similarity("患者姓名", "患者名")
print(f"Similarity: {similarity:.3f}")

# Batch similarity calculation
texts1 = ["患者姓名", "手术日期", "诊断结果"]
texts2 = ["患者名", "手术时间", "诊断"]
similarities = calculator.calculate_batch_similarity(texts1, texts2)
print(f"Batch similarities: {similarities}")
```

### Offline Mode

```python
# Use predefined similarities (no model download)
results = evaluate_entities(
    y_true, y_pred,
    mode="semi_structured",
    matching_method="text",
    text_match_mode="similarity",
    similarity_threshold=0.8,
    similarity_model="offline"  # No download required
)
```

## Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Symptoms**: Network timeout or download errors

**Solutions**:
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/
rm -rf ~/.cache/torch/sentence_transformers/

# Use offline mode as fallback
similarity_model="offline"
```

#### 2. Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
```python
# Use smaller model
similarity_model="small"  # ~100MB instead of ~500MB

# Or use offline mode
similarity_model="offline"  # No memory usage
```

#### 3. Slow Performance

**Symptoms**: Long processing times

**Solutions**:
```python
# Use smaller model for faster inference
similarity_model="tiny"  # Faster than "large"

# Reduce batch size
# (handled automatically by the library)
```

#### 4. Model Not Found

**Symptoms**: Model loading errors

**Solutions**:
```python
# Check available models
from src.similarity_utils import ChineseMedicalSimilarity
print(ChineseMedicalSimilarity.MODEL_OPTIONS)

# Use a different model
similarity_model="offline"  # Always available
```

### Environment-Specific Issues

#### Docker Containers

```dockerfile
# Set cache directory in Docker
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Create cache directory
RUN mkdir -p /app/cache
```

#### Cloud Environments

```python
# Set cache to persistent storage
import os
os.environ['TRANSFORMERS_CACHE'] = '/persistent/cache'
os.environ['HF_HOME'] = '/persistent/cache'
```

#### Air-Gapped Networks

```python
# Use offline mode for environments without internet
similarity_model="offline"

# Or pre-download models and copy to target environment
# Copy ~/.cache/huggingface/hub/ to target machine
```

## Performance Optimization

### Model Selection

- **Speed**: `"offline"` > `"small"` > `"tiny"` > `"medium"` > `"large"`
- **Accuracy**: `"large"` > `"medium"` > `"tiny"` > `"small"` > `"offline"`
- **Memory**: `"offline"` < `"small"` < `"tiny"` < `"medium"` < `"large"`

### Batch Processing

The library automatically handles batching for optimal performance:

```python
# Large datasets are automatically batched
results = evaluate_entities(
    large_y_true, large_y_pred,
    mode="semi_structured",
    matching_method="text",
    text_match_mode="similarity"
)
```

### Caching

Models are cached after first load for faster subsequent use:

```python
# First call: downloads and loads model
calculator1 = ChineseMedicalSimilarity(model_name="tiny")

# Second call: uses cached model
calculator2 = ChineseMedicalSimilarity(model_name="tiny")  # Fast
```

## Best Practices

### 1. Model Selection

- Start with `"tiny"` for most use cases
- Use `"medium"` for Chinese text if accuracy is critical
- Use `"large"` only for medical domain with sufficient resources
- Use `"offline"` for production environments without internet access

### 2. Threshold Tuning

```python
# Start with default threshold
similarity_threshold=0.8

# Adjust based on your data
# Higher threshold = more strict matching
# Lower threshold = more lenient matching
```

### 3. Error Handling

```python
try:
    results = evaluate_entities(
        y_true, y_pred,
        mode="semi_structured",
        matching_method="text",
        text_match_mode="similarity",
        similarity_model="tiny"
    )
except Exception as e:
    # Fallback to offline mode
    results = evaluate_entities(
        y_true, y_pred,
        mode="semi_structured",
        matching_method="text",
        text_match_mode="similarity",
        similarity_model="offline"
    )
```

### 4. Resource Management

```python
# For long-running processes, consider model cleanup
import gc
import torch

# After processing
del calculator
torch.cuda.empty_cache()  # If using GPU
gc.collect()
```

## Support

For issues related to model setup:

1. Check the troubleshooting section above
2. Verify your internet connection for model downloads
3. Ensure sufficient disk space for model storage
4. Check system memory for model loading
5. Use offline mode as a fallback option 