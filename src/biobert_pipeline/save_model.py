from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained(
    "models/biobert/checkpoints/checkpoint-654"
)

tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-v1.1"
)

model.save_pretrained(
    "models/biobert/final_model"
)

tokenizer.save_pretrained(
    "models/biobert/final_model"
)

print("Saved successfully")