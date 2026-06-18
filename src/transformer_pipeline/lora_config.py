from peft import LoraConfig, TaskType

'''
Tested for Rank = 4, 8, 16
'''

LORA_CONFIG = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=16,
    lora_alpha=32, # Keep 2*r
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
)

MODEL_NAME = (
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)

OUTPUT_DIR = "models/checkpoints/lora_r16"

SAVE_DIR = "models/pubmedbert_lora_r16"

TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "epochs": 5,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}