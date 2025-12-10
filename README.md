# NER Model with BERT and CRF implemented with Pytorch

This project implements a Named Entity Recognition (NER) model using BERT and CRF for Chinese medical text processing. The model is designed to extract structured information from medical reports.

## Architecture

The model follows a two-stage architecture:
1. BERT as the encoder to extract contextual features
2. CRF as the decoder to predict label sequences

```
Input Text -> BERT -> Linear Layer -> CRF -> Label Sequence
```

## Features

- Chinese medical text NER
- BERT-CRF architecture for improved sequence labeling
- Support for fine-tuning BERT layers
- Data preprocessing utilities
- Model training and evaluation scripts
- Data augmentation tools

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
ner-bert-crf/
├── ner/                 # Core NER model implementation
│   ├── model.py         # BERT-CRF model definition
│   ├── dataset.py       # Dataset processing utilities
│   ├── train.py         # Model training script
│   └── test.py          # Model testing/inference script
├── bc/                  # Breast cancer report processing
│   ├── res_stage1_inference.py  # Stage 1 inference
│   └── template/        # Templates and mappings
├── pre_struct/          # Data preprocessing and augmentation
│   ├── data_aug/        # Data augmentation pipeline
│   └── STRUCTURE_MAP_0919.json  # Medical report structure definitions
├── evaluation/          # Model evaluation tools
│   ├── src/             # Evaluation source code
│   └── tests/           # Evaluation tests
├── utils/               # Utility functions
├── config/              # Configuration files
├── data/                # Data files
└── README.md            # Project documentation
```

## Quick Start

### 1. Prepare Data

Prepare your training data in Label Studio format. The labeled data should be saved as `data/train_label.json`.

Label Studio export example:
```json
{
  "data": {
    "text": "患者张三，男，45岁，因胸痛就诊..."
  },
  "annotations": [{
    "result": [{
      "value": {
        "start": 2,
        "end": 4,
        "text": "张三",
        "labels": ["PATIENT_NAME"]
      }
    }]
  }]
}
```

### 2. Train the Model

```bash
python ner/train.py
```

### 3. Inference

```python
from ner.test import ner

s = "患者李四，女，38岁，因头痛发热3天就诊..."
result = ner(s)
print(result)
```

## Configuration

Key configuration parameters can be found and modified in:
- Model parameters: `config/model.conf`
- Label definitions: `config/labels.json`

## Data Augmentation

The project includes a comprehensive data augmentation pipeline in `pre_struct/data_aug/` that can:
- Synthesize new keys based on report structure
- Generate variations of existing records
- Apply noise injection for robustness

To run data augmentation:
```bash
python pre_struct/data_aug/pipeline.py
```

## Evaluation

Model performance can be evaluated using the tools in the `evaluation/` directory:

```bash
cd evaluation
python -m pytest tests/
```

## Model Details

The model uses a pre-trained multilingual BERT model as the backbone, with only the top 4 layers fine-tuned for the NER task. The CRF layer helps ensure valid label sequence predictions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cis_papers/159/)