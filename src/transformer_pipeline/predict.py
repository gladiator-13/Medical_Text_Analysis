import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

from config import ID2LABEL


MODEL_PATH = "models/biobert/final_model"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

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

    return list(zip(tokens, labels))


def main():

    text = (
        "Aspirin may cause gastric bleeding "
        "and ibuprofen can damage kidneys."
    )

    results = predict(text)

    for token, label in results:
        print(f"{token:15} {label}")

    entities = extract_entities(text)

    print("\nDetected Entities:\n")

    for entity, label in entities:
        print(f"{entity} --> {label}")

def extract_entities(text):

    predictions = predict(text)

    entities = []

    current_entity = []
    current_label = None

    for token, label in predictions:

        if token in ["[CLS]", "[SEP]"]:
            continue

        if label.startswith("B-"):

            if current_entity:
                entities.append(
                    (
                        " ".join(current_entity),
                        current_label
                    )
                )

            current_entity = [token]
            current_label = label[2:]

        elif (
            label.startswith("I-")
            and current_label == label[2:]
        ):

            current_entity.append(token)

        else:

            if current_entity:
                entities.append(
                    (
                        " ".join(current_entity),
                        current_label
                    )
                )

            current_entity = []
            current_label = None

    if current_entity:
        entities.append(
            (
                " ".join(current_entity),
                current_label
            )
        )

    return entities


if __name__ == "__main__":
    main()