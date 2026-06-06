import json
import spacy
import random # for shuffling training examples in each epoch for better training
from spacy.util import minibatch

from spacy.training import Example

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def create_spacy_pipeline():
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    return nlp

#The model architecture needs to know the possible output classes for training hence add_label func manually adds the labels
def add_labels(nlp, train_data):
    ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for start, end, label in annotations["entities"]:
            ner.add_label(label)

''' 
An Example contains BOTH:
- predicted doc state
- expected gold annotations

During training:
spaCy compares: prediction vs gold
and finds gradients
'''
def create_examples(nlp, train_data):
    examples = []

    for text,  annotations in train_data:
        doc = nlp.make_doc(text) #creates tokenized spaCy document without running the pipeline

        example = Example.from_dict(
            doc, annotations
        )

        examples.append(example)

    return examples

def train_ner_model(nlp, examples, epochs=50):
    optimizer = nlp.initialize() #Initializes model weights, internal tensors, optimizer state

    for epoch in range(epochs):
        random.shuffle(examples)

        losses = {}

        batches = minibatch(
            examples,
            size=8
        )

        for batch in batches:
            nlp.update(
                batch,
                drop=0.2,
                losses=losses,
                sgd=optimizer
            )
        # nlp.update - internally performs => forward pass, prediction, error computation, backpropagation, parameter update
        print(f"Epoch {epoch+1}")
        print(losses)

def inspect_predictions(nlp, validation_data, num_samples=5):
    for text, annotations in validation_data[:num_samples]:
        doc = nlp(text)

        print("\nTEXT: ")
        print(text)

        print("\nPREDICTED ENTITIES: ")
        for ent in doc.ents:
            print(f"{ent.text} -> {ent.label_}")

        print("\nGOLD ENTITIES: ")
        for start, end, label in annotations["entities"]:
            print(f"{text[start:end]} -> {label}")

        print("\n" + "=" * 60)

if __name__ == "__main__":
    train_data = load_data("data/preprocessed/train_spacy.json")
    validation_data = load_data("data/preprocessed/validation_spacy.json")

    nlp = create_spacy_pipeline()
    add_labels(nlp, train_data)
    # print(nlp.pipe_names)

    train_examples = create_examples(
        nlp, train_data
    )

    validation_examples = create_examples(
        nlp, validation_data
    )

    train_ner_model(nlp, train_examples, 25)

    #Evaluate on Validation set
    scores = nlp.evaluate(
        validation_examples
    )

    nlp.to_disk("models/biomedical_ner")
    print("Model saved successfully!")

    print(scores)
    
    inspect_predictions(nlp, validation_data)