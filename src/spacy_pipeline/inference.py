import spacy

nlp = spacy.load("models/biomedical_ner")

text = """A 52-year-old female stabilized on lithium for bipolar disorder 
presented with severe tremors and polyuria, which resolved upon 
switching to valproate; however, she subsequently developed thrombocytopenia.
"""

doc = nlp(text)
print("\nPREDICTED ENTITIES: \n")
for ent in doc.ents:
    print(f"{ent.text} --> {ent.label_}")