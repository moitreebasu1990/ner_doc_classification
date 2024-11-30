# NER Classification Project

A Named Entity Recognition (NER) and document type classification project that combines transformer-based models (BERT and T5) with Mistral's LLM for enhanced entity recognition and document classification. The project features both BERT and T5 architectures, with Mistral integration for intelligent document labeling.

## Features

- Dual model architecture support:
  - BERT-based NER model for sequence classification
  - T5-based model with encoder-only architecture for enhanced token-level predictions
- Mistral integration for intelligent document-level labeling
- Support for various entity types and document types
- Advanced training features:
  - Early stopping and model checkpointing
  - Configurable dropout rates
  - Optional encoder layer freezing (T5)
- Comprehensive visualization suite:
  - Learning curves
  - Training metrics
- Extensive test suite with pytest

## Requirements

- Python 3.11+
- Poetry for dependency management
- Mistral API key for document labeling

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ner_doc_classification
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

4. Set up your Mistral API key:
```bash
export PERSONAL_MISTRAL_API_KEY='your-api-key-here'
```

## Project Structure

```
ner_doc_classification/
├── ner_doc_classification/
│   ├── data/              # Dataset handling and preprocessing
│   │   ├── dataset.py     # Custom dataset implementations
│   │   └── data_utils.py  # Data loading and processing utilities
│   ├── models/            # Model architectures
│   │   ├── bert_ner.py    # BERT-based NER model
│   │   └── t5_advanced_ner.py  # T5-based NER model
│   ├── training/          # Training implementations
│   │   ├── train.py       # BERT training loop
│   │   └── train_t5.py    # T5 training loop
│   ├── utils/             # Utility functions
│   ├── visualization/     # Visualization tools
│   │   ├── visualization.py    # BERT visualizations
│   │   └── t5_visualization.py # T5 attention visualizations
│   ├── main.py           # BERT training pipeline
│   └── main_t5.py        # T5 training pipeline
├── tests/                # Test suite
├── pyproject.toml       # Project dependencies
└── README.md
```

## Usage

1. Activate the poetry environment:
```bash
poetry shell
```

2. Run training scripts:
```bash
# Train with BERT model
python -m ner_doc_classification.main

# Train with T5 model
python -m ner_doc_classification.main_t5
```

3. Run tests:
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=ner_doc_classification

# Run specific test files
pytest tests/test_models.py
```

4. Code quality:
```bash
# Format with black
black .

# Sort imports
isort .

# Run flake8
flake8 .
```

## Model Architectures

### BERT Model (Base)
The base BERT model implements a dual-task architecture for both NER and document classification:
- **Architecture**: BERT-base-uncased with additional classification heads
- **NER Head**: Token-level classification for entity recognition
- **Document Head**: Sequence-level classification using [CLS] token
- **Features**:
  - Shared BERT encoder for both tasks
  - Dropout regularization
  - Multi-task learning optimization

### Advanced T5 Model
The T5-based model incorporates several advanced features for enhanced performance:
- **Architecture**: T5 encoder-only with bilateral feature enhancement
- **Key Components**:
  - Bilateral Feature Enhancement: Captures both left and right context
  - Pre/Post Layer Normalization
  - Multi-level dropout strategy
  - Configurable attention patterns
- **Advanced Features**:
  - Adaptive layer freezing during training
  - Attention map visualization
  - Context-aware token representations

## Example Usage

For detailed examples and usage patterns, refer to our Jupyter notebooks [`ner_document_classification.ipynb`] in the `notebooks/` directory

## Training Process

1. Data Preparation:
   - Load CONLL2003 dataset
   - Generate document-level labels using Mistral
   - Split into train/validation sets

2. Model Training:
   - Configurable dataset sizes: [10, 30, 100, 300, 1000]
   - Early stopping based on validation loss
   - Learning rate scheduling
   - Gradient clipping

3. Visualization:
   - Training/validation loss curves

## License

This project is licensed under the MIT License - see the LICENSE file for details.