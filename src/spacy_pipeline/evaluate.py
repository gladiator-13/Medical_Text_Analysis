import json
import spacy
from spacy.training import Example

model_path = "models/spacy/biomedical_ner"
nlp = spacy.load(model_path)
print("Model loaded successfully!")

test_data_path = "data/preprocessed/test_spacy.json"

with open(test_data_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test samples.")

# -----------------------------------------------------------
# Convert test data to examples
# -----------------------------------------------------------
examples = []
for text, annotations in test_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    examples.append(example)

print(f"Created {len(examples)} evaluation examples.")

# -----------------------------------------------------------
# Evaluating the models performance
# -----------------------------------------------------------
scores = nlp.evaluate(examples)
print("\nEvaluation Completed!")

# --------------------------------------------------
# Extract metrics
# --------------------------------------------------
metrics = {
    "overall": {
        "precision": scores["ents_p"],
        "recall": scores["ents_r"],
        "f1": scores["ents_f"]
    },
    "per_entity": {}
}

# --------------------------------------------------
# Per-entity metrics
# --------------------------------------------------
for entity_label, entity_scores in scores["ents_per_type"].items():

    metrics["per_entity"][entity_label] = {
        "precision": entity_scores["p"],
        "recall": entity_scores["r"],
        "f1": entity_scores["f"]
    }

# --------------------------------------------------
# Save metrics to JSON
# --------------------------------------------------
OUTPUT_PATH = "results/metrics/spacy_metrics.json"

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(f"\nMetrics saved to: {OUTPUT_PATH}")

# --------------------------------------------------
# Print results
# --------------------------------------------------
print("\n========== OVERALL METRICS ==========")

print(f"Precision : {metrics['overall']['precision']:.4f}")
print(f"Recall    : {metrics['overall']['recall']:.4f}")
print(f"F1 Score  : {metrics['overall']['f1']:.4f}")

print("\n========== ENTITY-WISE METRICS ==========")

for entity, values in metrics["per_entity"].items():

    print(f"\nEntity: {entity}")

    print(f"Precision : {values['precision']:.4f}")
    print(f"Recall    : {values['recall']:.4f}")
    print(f"F1 Score  : {values['f1']:.4f}")