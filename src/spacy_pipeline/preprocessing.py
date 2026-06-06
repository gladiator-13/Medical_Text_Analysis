from datasets import load_dataset
from pprint import pprint #pretty print
from collections import Counter
import json

LABEL_MAP = {
    0: "O",
    1: "B-CHEMICAL", #Beginning of a chemical/drug name.
    2: "B-DISEASE", #Beginning of a disease entity.
    3: "I-DISEASE", #Continuation of a chemical entity.
    4: "I-CHEMICAL" #Continuation of disease entity.
}

def to_spacy(tokens, tags):
    text = " ".join(tokens)

    entities = []

    current_entity = None # Active entity text
    current_start = None # Character start index
    current_label = None # entity class

    char_position = 0 

    for token, tag in zip(tokens, tags):
        label = LABEL_MAP[tag]

        token_start = char_position
        token_end = char_position + len(token)

        if label.startswith("B-"):
            if current_entity is not None:
                entities.append(
                    (current_start, previous_end, current_label)
                )

            current_entity = token
            current_start = token_start
            current_label = label[2:]

        elif label.startswith("I-"):
            if current_entity is not None:
                current_entity += " " + token

        else:
            if current_entity is not None:
                entities.append(
                    (current_start, previous_end, current_label)
                )

            current_entity = None
            current_start = None
            current_label = None

        previous_end = token_end
        char_position = token_end + 1
    
    if current_entity is not None:
        entities.append(
            (current_start, previous_end, current_label)
        )

    return text, {"entities": entities}

def load_bc5cdr():
    dataset = load_dataset(
        "tner/bc5cdr"
    )

    print(dataset)

    return dataset

# def get_unique_tags(datasets):
#     unique_tags = set()

#     for sample in dataset["train"]:
#         unique_tags.update(sample["tags"])

#     return sorted(unique_tags)


# def count_tags(dataset):
#     tag_counter = Counter()

#     for sample in dataset["train"]:
#         tag_counter.update(sample["tags"])

#     return tag_counter

# def inspect_tag_examples(dataset, target_tag, num_examples=5):
#     found = 0

#     for sample in dataset["train"]:
#         tokens = sample["tokens"]
#         tags = sample["tags"]

#         for token, tag in zip(tokens, tags):
#             if tag == target_tag:
#                 print(f"Token: {token} | Tag: {tag}")
#                 found += 1

#                 if found >= num_examples:
#                     return
                
def convert_dataset_to_spacy(dataset_split):
    spacy_data = []

    for sample in dataset_split:
        text, annotation = to_spacy(
            sample["tokens"],
            sample["tags"]
        )

        spacy_data.append(
            (text, annotation) 
        )

    return spacy_data

# def validate_spacy_data(spacy_data, num_samples=5):
#     for text, annotation in spacy_data[:num_samples]:
#         print("\nTEXT: ")
#         print(text)

#         print("\nENTITIES: ")
#         for start, end, label in annotation["entities"]:
#             extracted_text = text[start:end]

#             print(
#                 f"{label}: {extracted_text} "
#                 f"{start}, {end}"
#             )

#         print("-"*50)

def save_spacy_data(spacy_data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            spacy_data,
            f, 
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved data to {output_path}")

'''encoding="utf-8"
Biomedical datasets may contain:
-symbols
-unicode
-special characters
UTF-8 avoids corruption.
--------------------------------------
indent=4
Makes JSON human-readable.
--------------------------------------
ensure_ascii=False
Prevents Unicode from becoming ugly escaped text.

'''
if __name__ == "__main__":
    dataset = load_bc5cdr()

    train_data = convert_dataset_to_spacy(
        dataset["train"]
    )
    save_spacy_data(
        train_data,
        "data/preprocessed/train_spacy.json"
    )

    validation_data = convert_dataset_to_spacy(
        dataset["validation"]
    )
    save_spacy_data(
        validation_data, 
        "data/preprocessed/validation_spacy.json"
    )

    test_data = convert_dataset_to_spacy(
        dataset["test"]
    )
    save_spacy_data(
        test_data,
        "data/preprocessed/test_pacy.json"
    )

    # validate_spacy_data(train_data)
    # print(train_data[0])

    
