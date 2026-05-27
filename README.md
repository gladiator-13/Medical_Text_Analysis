# Medical Text Analysis

Biomedical Named Entity Recognition (NER) system for extracting chemical and disease entities from biomedical literature using spaCy and the BC5CDR Dataset.

---

## Overview

This project builds a domain-specific NLP pipeline capable of identifying biomedical entities from clinical and research text.

The system:

- preprocesses biomedical datasets
- converts BIO-tagged annotations into spaCy-compatible spans
- trains a custom spaCy NER model
- evaluates entity-level performance
- performs inference on unseen medical text

The project was developed to explore:

- Biomedical NLP pipelines
- Named Entity Recognition (NER)
- BIO tagging schemes
- Sequence labeling
- NLP training dynamics
- Evaluation methodologies
- Modular ML system engineering

---

## Features

- Biomedical dataset preprocessing
- BIO tag parsing and span reconstruction
- Custom spaCy NER training pipeline
- Validation/test split handling
- Precision / Recall / F1 evaluation
- Minibatch optimization training
- Model persistence and loading
- Biomedical inference pipeline
- Modular project architecture

---

## Dataset

This project uses the BC5CDR biomedical dataset containing:

- biomedical research abstracts
- chemical entities
- disease entities
- token-level BIO annotations

### Example

#### Input

Naloxone reverses the antihypertensive effect of clonidine.

#### Entities

Naloxone  → CHEMICAL
clonidine → CHEMICAL

---

## Project Structure

Medical_Text_Analysis/
│
├── data/
│   └── processed/
│
├── models/
│
├── notebooks/
│   └── Medical_ner_exploration.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── inference.py
│
├── requirements.txt
├── README.md
└── .gitignore

---

# Installation

## 1. Clone the Repository

git clone <your-repository-link>
cd Medical_Text_Analysis

---

## 2. Create a Virtual Environment

### Windows

python -m venv venv
venv\Scripts\activate

### Linux / macOS

python3 -m venv venv
source venv/bin/activate

---

## 3. Install Dependencies

pip install -r requirements.txt

---

# Preprocessing Pipeline

The preprocessing pipeline:

- loads the BC5CDR dataset
- converts BIO tags into entity spans
- validates annotations
- creates train/validation/test splits
- exports processed data into JSON format

Run preprocessing:

python src/preprocessing.py

Processed files are stored inside:

data/preprocessed/

---

# Model Training

Train the biomedical NER model:

python src/train.py

Training includes:

- spaCy pipeline initialization
- label registration
- minibatch optimization
- validation evaluation
- model checkpoint persistence

---

# Model Evaluation

## Validation Performance

| Metric | Score |
|---|---|
| Entity Precision | ~0.83 |
| Entity Recall | ~0.77 |
| Entity F1 Score | ~0.80 |

---

## Entity-wise Performance

| Entity Type | F1 Score |
|---|---|
| CHEMICAL | ~0.84 |
| DISEASE | ~0.75 |

---

# Example Prediction

## Input

Lithium carbonate toxicity caused congestive heart failure.

## Output

Lithium carbonate         → CHEMICAL
toxicity                  → DISEASE
congestive heart failure  → DISEASE

---

# Inference

Run inference using the trained model:

python src/predict.py

Example usage:

import spacy

nlp = spacy.load("models/biomedical_ner")

text = "Aspirin toxicity caused cardiac arrest."

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)

---

# Concepts Explored

This project explores:

- Named Entity Recognition (NER)
- Biomedical NLP
- BIO tagging schemes
- Span boundary prediction
- Precision / Recall tradeoffs
- Minibatch optimization
- Model persistence
- NLP pipeline engineering

---

# Future Improvements

Potential future extensions include:

- Transformer-based biomedical NER
- BioBERT integration
- PubMedBERT fine-tuning
- Biomedical relation extraction
- Entity linking
- Clinical document processing
- Error analysis dashboards
- Streamlit/Gradio deployment

---

# Technologies Used

- Python
- spaCy
- Hugging Face Datasets
- NumPy

---

# Learning Outcomes

Through this project:

- biomedical NLP pipelines were implemented from scratch
- custom NER systems were trained and evaluated
- training instability and optimization behavior were analyzed
- modular ML system design principles were applied
- classical NLP limitations were explored before transitioning to transformer architectures

---

# License

This project is intended for educational and research purposes.
