from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification )

from transformers import AutoTokenizer
from dataset import get_tokenized_dataset, tokenizer
from config import ID2LABEL, LABEL2ID
from metrics import compute_metrics

models = {
   "biobert" : "dmis-lab/biobert-v1.1",
   "pubmed" : "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
}

def main():

    dataset = get_tokenized_dataset()

    model = AutoModelForTokenClassification.from_pretrained(
        models["biobert"],
        num_labels=5,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir="models/biobert/checkpoints", #models/pubmed_bert/MODEL_NAME => for pubmed model

        num_train_epochs=7,

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        logging_steps=50,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],

        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Trainer initialized successfully")
    trainer.train()

    trainer.save_model(
      "models/biobert/model_7epoch" #models/pubmed_bert/MODEL_NAME
    )

    tokenizer.save_pretrained(
      "models/biobert/model_7epoch" #models/pubmed_bert/MODEL_NAME
    )


if __name__ == "__main__":
  main()
    
