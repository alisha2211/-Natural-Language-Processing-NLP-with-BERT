# -Natural-Language-Processing-NLP-with-BERT


from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset

# Load dataset
dataset = load_dataset("glue", "mrpc")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], padding=True, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up Trainer
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train and evaluate the model
trainer.train()
