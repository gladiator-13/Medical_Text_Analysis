# Biomedical NER on BC5CDR: spaCy vs BioBERT vs PubMedBERT + LoRA

Comparing classical and transformer-based Named Entity Recognition for extracting **Chemical** and **Disease** entities from biomedical literature using the BC5CDR benchmark dataset, including a study of LoRA parameter-efficient fine-tuning.

---

## Key Findings

- **Both transformer models exceed published BioBERT baselines** — BioBERT achieves 94.00% Chemical F1 vs published 93.47%, and PubMedBERT achieves 94.59%, on clean independent implementations
- **Disease entities are consistently harder than Chemical entities** — across all models and configurations, Disease F1 lags Chemical F1 by 8-12 points, suggesting disease names are more context-dependent and harder to identify from surface form alone
- **Purer domain pretraining helps more on harder entity types** — PubMedBERT improves Disease F1 by +1.57 points over BioBERT but Chemical F1 by only +0.59 points, confirming domain specificity matters most where the task is hardest
- **PubMedBERT converges faster** — plateaus at 5 epochs vs BioBERT's 7 epochs, despite achieving higher final performance
- **LoRA retains strong Chemical performance at any rank** — Chemical F1 stays above 93.2% across all LoRA configurations (vs 94.59% full fine-tuning), a drop of only ~1.3 points with 98%+ parameter reduction
- **Disease is disproportionately sensitive to parameter reduction** — Disease F1 drops 3.29 points at r=4 vs only 1.32 points on Chemical, confirming harder entities need more model capacity
- **Rank matters almost exclusively for Disease** — increasing rank from 4→16 improves Disease F1 by +1.95 points but Chemical F1 by only +0.03 points
- **Transformers reduce total errors by 60%+** — false negatives drop from 1040→226 on Chemical and 655→187 on Disease; however Disease false positives actually increased from spaCy (383) to transformers (~415), indicating over-prediction of disease-like phrases

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

### LoRA Fine-Tuning vs Full Fine-Tuning (PubMedBERT base)

| Method | Trainable Params | Chemical F1 | Disease F1 | Overall F1 | vs Full FT |
|--------|-----------------|-------------|------------|------------|------------|
| Full fine-tune | 100% | 94.59% | 85.70% | 90.64% | — |
| LoRA r=4 (5 epochs) | ~0.4% | 93.24% | 80.46% | 87.49% | -3.15 |
| LoRA r=4 (7 epochs) | ~0.4% | 93.42% | 81.20% | 87.95% | -2.69 |
| LoRA r=8 (5 epochs) | ~0.8% | 93.24% | 81.50% | 87.97% | -2.67 |
| LoRA r=16 (5 epochs) | ~1.5% | 93.27% | 82.41% | 88.42% | -2.22 |

### Full Model Comparison vs Published Baselines

| Model | Chemical F1 | Disease F1 | Overall F1 |
|-------|-------------|------------|------------|
| spaCy (baseline) | 81.73% | 75.41% | 78.81% |
| BioBERT (7 epochs) | 94.00% | 84.13% | 89.61% |
| **PubMedBERT (5 epochs)** | **94.59%** | **85.70%** | **90.64%** |
| LoRA r=16 (best LoRA) | 93.27% | 82.41% | 88.42% |
| Published SciSpaCy *(Sohail et al., 2024)* | — | — | 85.53% |
| Published BioBERT *(Sohail et al., 2024)* | — | — | 87.83% |
| Published BioBERT v1.1 *(Lee et al., 2020)* | 93.47% | 87.15% | — |
| BioALBERT *(Naseem et al., 2021)* | 97.90% | 97.66% | — |

> **References:** Lee et al. (2020) *BioBERT: a pre-trained biomedical language representation model*; Sohail et al. (2024) *Exploring Biomedical Named Entity Recognition via SciSpaCy and BioBERT Models*; Naseem et al. (2021) *BioALBERT: A Simple and Effective Pre-trained Language Model for Biomedical NER*

---

## Error Analysis

Errors were categorized into four types across all three models on the BC5CDR test set — boundary errors (correct entity type, wrong span), overlap errors (nested or overlapping spans), false positives (predicted where no entity exists), and false negatives (missed entities entirely).

### Error Counts by Model and Entity Type

| Error Type | spaCy Chem | BioBERT Chem | PubMedBERT Chem | spaCy Dis | BioBERT Dis | PubMedBERT Dis |
|---|---|---|---|---|---|---|
| Boundary Errors | 140 | 120 | 96 | 457 | 370 | 341 |
| Overlap Errors | 116 | 14 | 22 | 55 | 19 | 10 |
| False Positives | 336 | 295 | 284 | 383 | 422 | 415 |
| False Negatives | 1040 | 243 | 226 | 655 | 245 | 187 |
| **Total** | **1632** | **672** | **628** | **1550** | **1056** | **953** |

### Analysis

Transformer models reduce total errors by over 60% compared to spaCy, with false negatives showing the sharpest decline — from 1040 to 226 on Chemical and 655 to 187 on Disease — indicating that biomedical pretraining dramatically improves entity recall. Overlap errors, a spaCy-specific failure mode (116 on Chemical), nearly disappear with transformers (14–22), showing attention-based models handle entity span boundaries far more cleanly.

However, false positives proved resistant to improvement across all models. Disease false positives actually increased from spaCy (383) to BioBERT (422) and PubMedBERT (415), suggesting transformers over-predict disease-like phrases due to broader pattern recognition. Boundary errors on Disease entities remained persistently high (341 in PubMedBERT vs 96 for Chemical), indicating that disease span delimitation is a fundamental challenge not resolved by domain-specific pretraining alone. These findings motivated the LoRA experiments and suggest future work on CRF decoding layers or boundary-aware training objectives.

---

## Overview

This project builds a complete biomedical NER pipeline — from data preprocessing through model training, evaluation, error analysis, and parameter-efficient fine-tuning — comparing a classical spaCy-based approach against fine-tuned transformer models on the BC5CDR benchmark.

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
│   ├── spacy/
│   ├── lora/
│   ├── biobert/
│   └── pubmedbert/
│
├── notebooks/
│   ├── biobert_pipeline.ipynb
│   ├── biobert_training.ipynb
│   ├── error_analysis.ipynb
│   ├── medical_ner_exploration.ipynb
│   └── lora_experiments.ipynb
│
├── results/
│   ├── spacy_metrics/
│   │   └── spacy_metrics.json
│   ├── biobert_metrics/
│   │   ├── model_3epochs.json
│   │   ├── model_5epochs.json
│   │   ├── model_7epochs.json
│   │   └── experiments.csv
│   ├── pubmedbert_metrics/
│   │   ├── model_3epochs.json
│   │   ├── model_5epochs.json
│   │   └── experiments.csv
│   ├── lora_metrics/
│   │   ├── lora_rank4_epoch5.json
│   │   ├── lora_rank4_epoch7.json
│   │   ├── lora_rank8_epoch5.json
│   │   ├── lora_rank16_epoch5.json
│   │   └── lora_experiments.csv
│   └── error_analysis/
│       ├── spacy_errors.json
│       ├── biobert_errors.json
│       └── pubmedbert_errors.json
│
├── src/
│   ├── common/
│   │   ├── utils.py
│   │   └── error_analysis.py
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
│       ├── metrics.py
│       ├── lora_config.py
│       └── lora_train.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/gladiator-13/Medical_Text_Analysis
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

### Transformer Pipeline (BioBERT / PubMedBERT)

```bash
# Fine-tune on BC5CDR
python -m src.transformer_pipeline.train

# Evaluate on test set
python -m src.transformer_pipeline.evaluate_model

# Run inference on new text
python -m src.transformer_pipeline.predict
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

All runs logged to CSV files per model in `results/`. Each run records: model, epochs, precision, recall, F1, accuracy, and entity-wise F1. Per-run detailed metrics stored as JSON files alongside each CSV.

---

## Stack

- Python
- PyTorch
- HuggingFace Transformers + Datasets + PEFT
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
- [x] Error analysis — characterising Chemical vs Disease failure modes
- [x] LoRA parameter-efficient fine-tuning (rank 4 / 8 / 16)

---

## License

For educational and research purposes.