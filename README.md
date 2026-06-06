# Biomedical NER on BC5CDR: spaCy vs BioBERT

Comparing classical and transformer-based Named Entity Recognition for extracting **Chemical** and **Disease** entities from biomedical literature using the BC5CDR benchmark dataset.

---

## Key Findings

- **BioBERT nearly matches published SOTA on Chemical entities** — 94.00% F1 vs published 94.40%, a gap of only 0.40 points
- **Disease entities are significantly harder** — 84.13% F1 vs published 89.90%, a gap of 5.77 points, suggesting disease names are more context-dependent and harder to detect from surface form alone
- **Domain-specific pretraining matters** — BioBERT outperforms a spaCy baseline substantially on both entity types
- **Performance plateaus around epoch 7** — gains reduce from +2.16 F1 (epochs 3→5) to +0.50 F1 (epochs 5→7), indicating diminishing returns beyond this point

---

## Results

### Epoch-wise BioBERT Performance

| Epochs | Precision | Recall | Overall F1 | Chemical F1 | Disease F1 |
|--------|-----------|--------|------------|-------------|------------|
| 3 | 84.61% | 89.43% | 86.95% | 92.46% | 80.17% |
| 5 | 87.80% | 90.46% | 89.11% | 93.61% | 83.54% |
| **7** | **88.40%** | **90.87%** | **89.61%** | **94.00%** | **84.13%** |

### Model Comparison

| Model | Chemical F1 | Disease F1 | Overall F1 |
|-------|-------------|------------|------------|
| spaCy (baseline) | 81.73% | 75.41% | 78.81% |
| BioBERT (7 epochs) | 94.00% | 84.13% | 89.61% |
| Published BioBERT | ~93.85% | ~89.16% | ~91.51% |
| Published SOTA (PubMedBERT) | 94.40% | 89.90% | 92.15% |

> Published numbers from the [BC5CDR leaderboard on Papers With Code](https://paperswithcode.com/sota/named-entity-recognition-on-bc5cdr)

---

## Overview

This project builds a complete biomedical NER pipeline — from data preprocessing through model training and evaluation — and compares a classical spaCy-based approach against fine-tuned BioBERT on the BC5CDR benchmark.

The BC5CDR dataset contains biomedical research abstracts annotated with two entity types:
- **Chemical** — drug names, chemical compounds (e.g. "acetaminophen", "sodium chloride")
- **Disease** — disease and condition names (e.g. "hypertension", "acute respiratory failure")

---

## Project Structure

```
Medical_Text_Analysis/
│
├── data/
│   └── preprocessed/
│       └── spacy/
│
├── models/
│   └── biobert/
│
├── notebooks/
│   ├── biobert_pipeline.ipynb
│   └── biobert_training.ipynb
│
├── results/
│   ├── spacy_metrics/
│   │   └── spacy_metrics.json
│   └── biobert_metrics/
│       ├── model_1epoch.json
│       ├── model_3epochs.json
│       ├── model_5epochs.json
│       ├── model_7epochs.json
│       └── experiments.csv
│
├── src/
│   ├── common/
│   │   └── utils.py
│   ├── spacy_pipeline/
│   │   ├── preprocessing.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   └── biobert_pipeline/
│       ├── config.py
│       ├── dataset.py
│       ├── train.py
│       ├── save_model.py
│       ├── evaluate_model.py
│       ├── save_metrics.py
│       ├── predict.py
│       └── metrics.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone <repository-url>
cd Medical_Text_Analysis
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### spaCy Pipeline

```bash
# Preprocess data
python src/spacy_pipeline/preprocessing.py

# Train spaCy NER model
python src/spacy_pipeline/train.py

# Evaluate on test set
python src/spacy_pipeline/evaluate.py

# Run inference on new text
python src/spacy_pipeline/inference.py
```

### BioBERT Pipeline

```bash
# Fine-tune BioBERT on BC5CDR
python -m src/biobert_pipeline/train.py

# Evaluate on test set
python -m src/biobert_pipeline/evaluate_model.py

# Run inference on new text
python -m src/biobert_pipeline/predict.py
```

---

## Dataset

**BC5CDR (BioCreative V Chemical-Disease Relation)**

- Biomedical research abstracts with token-level BIO annotations
- Two entity types: Chemical, Disease
- Standard train / validation / test splits
- Loaded via HuggingFace Datasets: `drAbreu/biocreative_NLP_BC5CDR`

---

## Experiment Tracking

All runs are logged to:

```
results/biobert_metrics/experiments.csv
```

Each run records: epochs, precision, recall, F1, accuracy.

Per-run detailed metrics (overall + entity-wise) stored as JSON in `results/biobert_metrics/`.

---

## Stack

- Python
- PyTorch
- HuggingFace Transformers + Datasets
- BioBERT (`dmis-lab/biobert-v1.1`)
- spaCy
- SeqEval
- NumPy

---

## Roadmap

- [ ] PubMedBERT fine-tuning and comparison
- [ ] LoRA parameter-efficient fine-tuning (rank 4 / 8 / 16)
- [ ] Error analysis — characterising Chemical vs Disease failure modes
- [ ] Relation extraction (Chemical–Disease pairs)
- [ ] Gradio inference demo

---

## License

For educational and research purposes.