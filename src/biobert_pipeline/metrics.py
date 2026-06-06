import evaluate

seqeval = evaluate.load("seqeval")

LABELS = [
    "O",
    "B-CHEMICAL",
    "B-DISEASE",
    "I-DISEASE",
    "I-CHEMICAL"
]

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = predictions.argmax(axis=2)

    true_predictions = [
        [
            LABELS[p]
            for p, l in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [
            LABELS[l]
            for p, l in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }