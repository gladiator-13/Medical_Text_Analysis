from typing import Dict, List
from pathlib import Path
import json
import spacy
# from ..transformer_pipeline.config import ID2LABEL, LABEL2ID
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
import torch

LABELS = {
    0: "O",
    1: "B-CHEMICAL",
    2: "B-DISEASE",
    3: "I-DISEASE",
    4: "I-CHEMICAL"
}

ID2LABEL = LABELS

LABEL2ID = {
    v: k
    for k, v in LABELS.items()
}

def overlap(start1, end1, start2, end2):
    return max(start1, start2) < min(end1, end2)


def analyze_errors(texts, gold_entities, predicted_entities):

    stats = {
        "CHEMICAL": {
            "boundary_errors": 0,
            "overlap_errors": 0,
            "false_positives": 0,
            "false_negatives": 0
        },
        "DISEASE": {
            "boundary_errors": 0,
            "overlap_errors": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
    }

    error_examples = {
        "CHEMICAL": [],
        "DISEASE": []
    }

    for sentence, gold, pred in zip(
        texts,
        gold_entities,
        predicted_entities
    ):

        gold_set = set(tuple(entity) for entity in gold)
        pred_set = set(tuple(entity) for entity in pred)

        exact_matches = gold_set & pred_set

        unmatched_gold = gold_set - exact_matches
        unmatched_pred = pred_set - exact_matches

        used_predictions = set()

        # Analyze unmatched gold entities
        for gold_entity in unmatched_gold:

            g_start, g_end, g_label = gold_entity

            found_match = False

            for pred_entity in unmatched_pred:

                # Prevent one prediction from matching multiple gold entities
                if pred_entity in used_predictions:
                    continue

                p_start, p_end, p_label = pred_entity

                if overlap(
                    g_start,
                    g_end,
                    p_start,
                    p_end
                ):

                    found_match = True
                    used_predictions.add(pred_entity)

                    error_record = {
                        "sentence": sentence,
                        "gold_entity": sentence[g_start:g_end],
                        "pred_entity": sentence[p_start:p_end],
                        "gold_label": g_label,
                        "pred_label": p_label,
                        "error": ""
                    }

                    if g_label == p_label:

                        stats[g_label]["boundary_errors"] += 1
                        error_record["error"] = "Boundary Error"

                    else:

                        stats[g_label]["overlap_errors"] += 1
                        error_record["error"] = (
                            "Overlap/Ambiguity Error"
                        )

                    error_examples[g_label].append(
                        error_record
                    )

                    break

            # False Negative
            if not found_match:

                stats[g_label]["false_negatives"] += 1

                error_examples[g_label].append(
                    {
                        "sentence": sentence,
                        "gold_entity": sentence[g_start:g_end],
                        "pred_entity": None,
                        "gold_label": g_label,
                        "pred_label": None,
                        "error": "False Negative"
                    }
                )

        # False Positives
        for pred_entity in unmatched_pred:

            if pred_entity in used_predictions:
                continue

            p_start, p_end, p_label = pred_entity

            if p_label not in stats:
                continue

            stats[p_label]["false_positives"] += 1

            error_examples[p_label].append(
                {
                    "sentence": sentence,
                    "gold_entity": None,
                    "pred_entity": sentence[p_start:p_end],
                    "gold_label": None,
                    "pred_label": p_label,
                    "error": "False Positive"
                }
            )

    return {
        "summary": stats,
        "examples": error_examples
    }


def spacy_dataset():

    test_data_path = "data/preprocessed/spacy/test_spacy.json"

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    model_path = Path("models/spacy/biomedical_ner")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path.resolve()}"
        )

    nlp = spacy.load(str(model_path))

    texts = []
    gold_entities = []
    predicted_entities = []

    for text, annotation in test_data:

        texts.append(text)
        gold_entities.append(annotation["entities"])

        doc = nlp(text)

        entities = []

        for ent in doc.ents:
            entities.append([
                ent.start_char,
                ent.end_char,
                ent.label_
            ])

        predicted_entities.append(entities)

    # print(f"Total Sentences: {len(texts)}")
    # print(f"Gold Entity Lists: {len(gold_entities)}")
    # print(f"Prediction Lists: {len(predicted_entities)}")

    result = analyze_errors(texts, gold_entities, predicted_entities)

    output_dir = Path("results/error_analysis")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = output_dir / "spacy_errors.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nSaved to: {output_file}")

def transformer_dataset():
    test_data_path = "data/preprocessed/spacy/test_spacy.json"

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    texts = []
    gold_entities = []

    for text, annotation in test_data:
        texts.append(text)
        gold_entities.append(annotation["entities"])

    tokenizer = AutoTokenizer.from_pretrained(
        "dmis-lab/biobert-v1.1"
    )
    model = AutoModelForTokenClassification.from_pretrained(
        "models/biobert/model_7epoch"
    )
    model.eval()

    def predict(text):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True
        )

        offset_mapping = inputs.pop("offset_mapping")

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(
            outputs.logits,
            dim=2
        )

        tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

        labels = [
            ID2LABEL[label.item()]
            for label in predictions[0]
        ]

        offsets = offset_mapping[0]

        return list(zip(tokens, labels, offsets))

    def extract_entities(text):

        predictions = predict(text)

        entities = []

        current_start = None
        current_end = None
        current_label = None

        for token, label, offset in predictions:

            start, end = offset.tolist()

            if token in ["[CLS]", "[SEP]"]:
                continue

            if start == end:
                continue

            if label.startswith("B-"):

                if current_start is not None:

                    entities.append(
                        [
                            current_start,
                            current_end,
                            current_label
                        ]
                    )

                current_start = start
                current_end = end
                current_label = label[2:]

            elif (
                label.startswith("I-")
                and current_label == label[2:]
            ):

                current_end = end

            else:

                if current_start is not None:

                    entities.append(
                        [
                            current_start,
                            current_end,
                            current_label
                        ]
                    )

                current_start = None
                current_end = None
                current_label = None

        if current_start is not None:

            entities.append(
                [
                    current_start,
                    current_end,
                    current_label
                ]
            )

        return entities
    
    predicted_entities = []

    for i in range(len(test_data)):
        text = test_data[i][0]
        predicted_entities.append(extract_entities(text))

    result = analyze_errors(texts, gold_entities, predicted_entities)

    output_dir = Path("results/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "biobert_errors.json"

    with open(output_file, "w", encoding="utf-8", ) as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":

    # spacy_dataset()
    transformer_dataset()

    # print("\nError Analysis Summary:")
    # print(json.dumps(results["summary"], indent=4))

    

