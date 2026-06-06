from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-v1.1"
)

label_map = {
    1: 4,
    2: 3
}

def load_bc5cdr():
    return load_dataset("tner/bc5cdr")


def tokenize_and_align_labels(example):

    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True
    )

    word_ids = tokenized.word_ids()

    labels = []
    previous_word_id = None

    for word_id in word_ids:

        if word_id is None:
            labels.append(-100)

        elif word_id != previous_word_id:
            labels.append(
                example["tags"][word_id]
            )

        else:
            label = example["tags"][word_id]

            if label in label_map:
                label = label_map[label]

            labels.append(label)

        previous_word_id = word_id

    tokenized["labels"] = labels

    return tokenized

def get_tokenized_dataset():

    dataset = load_bc5cdr()

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels
    )

    # tokenized_dataset.set_format(
    #     type="torch",
    #     columns=[
    #         "input_ids",
    #         "attention_mask",
    #         "labels"
    #     ]
    # )

    return tokenized_dataset

# dataset = get_tokenized_dataset()

# print(dataset["train"][0].keys())

# print(
#     len(dataset["train"][0]["input_ids"]),
#     len(dataset["train"][0]["labels"])
# )