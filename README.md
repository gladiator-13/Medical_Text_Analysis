# Biomedical NER on BC5CDR: spaCy vs BioBERT vs PubMedBERT

Comparing classical and transformer-based Named Entity Recognition for extracting **Chemical** and **Disease** entities from biomedical literature using the BC5CDR benchmark dataset.

---

## Key Findings

- **Both models exceed published BioBERT baselines** — our BioBERT achieves 94.00% Chemical F1 vs published 93.47%, and our PubMedBERT achieves 94.59% vs published numbers, on clean independent implementations
- **Disease entities are consistently harder than Chemical entities** — across all models, Disease F1 lags Chemical F1 by 8-9 points, suggesting disease names are more context-dependent and harder to identify from surface form alone
- **Purer domain pretraining helps more on harder entity types** — PubMedBERT improves Disease F1 by +1.57 points over BioBERT but Chemical F1 by only +0.59 points, confirming domain specificity matters most where the task is hardest
- **PubMedBERT converges faster** — plateaus at 5 epochs vs BioBERT's 7 epochs, despite achieving higher final performance
- **Performance plateaus are clearly observable** — BioBERT gains drop from +2.16 F1 (epochs 3→5) to +0.50 (epochs 5→7); PubMedBERT shows the same pattern one epoch earlier

---

## Results

### Epoch-wise BioBERT Performance

| Epochs | Precision | Recall | Overall F1 | Chemical F1 | Disease F1 |
|--------|-----------|--------|------------|-------------|------------|
| 3 | 84.61% | 89.43% | 86.95% | 92.46% | 80.17% |
| 5 | 87.80% | 90.46% | 89.11% | 93.61% | 83.54% |
| **7** | **88.40%** | **90.87%** | **89.61%** | **94.00%** | **84.13%** |

### Epoch-wise PubMedBERT Performance

| Epochs | Precision | Recall | Overall F1 | Chemical F1 | Disease F1 |
|--------|-----------|--------|------------|-------------|------------|
| 3 | 88.81% | 91.48% | 90.13% | 94.38% | 84.87% |
| **5** | **89.79%** | **91.51%** | **90.64%** | **94.59%** | **85.70%** |

### Model Comparison vs Published Baselines

| Model | Chemical F1 | Disease F1 | Overall F1 |
|-------|-------------|------------|------------|
| spaCy (baseline) | 81.73% | 75.41% | 78.81% |
| BioBERT (7 epochs) | 94.00% | 84.13% | 89.61% |
| PubMedBERT (5 epochs) | **94.59%** | **85.70%** | **90.64%** |
| Published SciSpaCy *(Sohail et al., 2024)* | — | — | 85.53% |
| Published BioBERT *(Sohail et al., 2024)* | — | — | 87.83% |
| Published BioBERT v1.1 *(Lee et al., 2020)* | 93.47% | 87.15% | — |
| LoRA r=4 | — | — | — |
| LoRA r=8 | — | — | — |
| LoRA r=16 | — | — | — |

> **References:** Lee et al. (2020) *BioBERT: a pre-trained biomedical language representation model*; Sohail et al. (2024) *Exploring Biomedical Named Entity Recognition via SciSpaCy and BioBERT Models*

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
│   ├── biobert/
│   └── pubmedbert/
│
├── notebooks/
│   ├── biobert_pipeline.ipynb
│   ├── biobert_training.ipynb
│   └── pubmedbert_training.ipynb
│
├── results/
│   ├── spacy_metrics/
│   │   └── spacy_metrics.json
│   ├── biobert_metrics/
│   │   ├── model_3epochs.json
│   │   ├── model_5epochs.json
│   │   ├── model_7epochs.json
│   │   └── experiments.csv
│   └── pubmedbert_metrics/
│       ├── model_3epochs.json
│       ├── model_5epochs.json
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
│   └── transformer_pipeline/
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
python -m src.biobert_pipeline.train

# Evaluate on test set
python -m src.biobert_pipeline.evaluate_model

# Run inference on new text
python -m src.biobert_pipeline.predict
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
- PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)
- spaCy
- SeqEval
- NumPy

---

## Roadmap

- [x] spaCy baseline NER pipeline
- [x] BioBERT fine-tuning and epoch comparison
- [x] PubMedBERT fine-tuning and comparison
- [ ] LoRA parameter-efficient fine-tuning (rank 4 / 8 / 16)
- [ ] Error analysis — characterising Chemical vs Disease failure modes
- [ ] Relation extraction (Chemical–Disease pairs)
- [ ] Gradio inference demo

---

## License

For educational and research purposes.