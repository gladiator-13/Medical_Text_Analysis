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