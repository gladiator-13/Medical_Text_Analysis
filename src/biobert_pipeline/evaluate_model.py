from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    DataCollatorForTokenClassification
)

from dataset import get_tokenized_dataset
from metrics import compute_metrics
import pandas as pd
from pathlib import Path
import json
import numpy as np
from config import ID2LABEL
import evaluate

def main():

    dataset = get_tokenized_dataset()

    model = AutoModelForTokenClassification.from_pretrained(
        "models/biobert/model_7epoch"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "models/biobert/model_7epoch"
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model=model,
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate()

    record = {
        "epochs": 7,
        "learning_rate": 2e-5,
        "batch_size": 8,
        "precision": results["eval_precision"],
        "recall": results["eval_recall"],
        "f1": results["eval_f1"]
    }

    file = Path("results/biobert_metrics/experiments.csv")

    if file.exists() and file.stat().st_size > 0:
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(file, index=False)

    # print(results)

    predictions, labels, _ = trainer.predict(
        dataset["validation"]
    )

    predictions = np.argmax(
        predictions,
        axis=2
    )

    #Computing Detailed Metrics
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):

        current_predictions = []
        current_labels = []

        for pred, lab in zip(prediction, label):

            if lab != -100:

                current_predictions.append(
                    ID2LABEL[pred]
                )

                current_labels.append(
                    ID2LABEL[lab]
                )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    seqeval = evaluate.load("seqeval")

    detailed_metrics = seqeval.compute(
        predictions=true_predictions,
        references=true_labels
    )

    #Building JSON
    metrics = {
        "overall": {
            "precision": float(
                detailed_metrics["overall_precision"]
            ),
            "recall": float(
                detailed_metrics["overall_recall"]
            ),
            "f1": float(
                detailed_metrics["overall_f1"]
            ),
            "accuracy": float(
                detailed_metrics["overall_accuracy"]
            )
        },
        "per_entity": {}
    }

    #Extracting Entity Metrics
    for key, value in detailed_metrics.items():

        if key.startswith("overall"):
            continue

        metrics["per_entity"][key] = {
            "precision": float(
                value["precision"]),

            "recall": float(
                value["recall"]),

            "f1": float(
                value["f1"]),

            "support": float(
                value["number"])
        }

    #Saving JSON
    OUTPUT_PATH = (
        "results/biobert_metrics/"
        "model_7epochs.json"
    )

    for key, value in metrics["overall"].items():
        print(
            "overall",
            key,
            value,
            type(value)
        )

        for entity, entity_metrics in metrics["per_entity"].items():

            print("\nENTITY:", entity)

            for key, value in entity_metrics.items():

                print(
                    key,
                    value,
                    type(value)
                )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(
        f"Metrics saved to "
        f"{OUTPUT_PATH}"
    )

if __name__ == "__main__":
    main()