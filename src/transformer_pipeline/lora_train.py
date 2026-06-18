from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from peft import get_peft_model

from dataset import get_tokenized_dataset
from metrics import compute_metrics

from config import (
    ID2LABEL,
    LABEL2ID,
)

from lora_config import (
    LORA_CONFIG,
    MODEL_NAME,
    OUTPUT_DIR,
    SAVE_DIR,
    TRAINING_CONFIG,
)

dataset = get_tokenized_dataset()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# Load Base Model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(ID2LABEL),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# Attach LoRA
lora_model = get_peft_model(
    model,
    LORA_CONFIG
)

lora_model.print_trainable_parameters()

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    learning_rate=TRAINING_CONFIG["learning_rate"],
    num_train_epochs=TRAINING_CONFIG["epochs"],

    per_device_train_batch_size=
        TRAINING_CONFIG["train_batch_size"],

    per_device_eval_batch_size=
        TRAINING_CONFIG["eval_batch_size"],

    warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
    weight_decay=TRAINING_CONFIG["weight_decay"],

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="eval_f1",
    greater_is_better=True,

    logging_steps=50,
    seed=42,
)

#Trainer
data_collator = DataCollatorForTokenClassification(
    tokenizer
)

trainer = Trainer(
    model=lora_model,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,
    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# Train and Save
trainer.train()

lora_model.save_pretrained(
    SAVE_DIR
)

tokenizer.save_pretrained(
    SAVE_DIR
)